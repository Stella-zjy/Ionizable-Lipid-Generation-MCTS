import torch
import torch.nn.functional as F
import numpy as np
import random
import math
import json
import itertools
from tqdm import tqdm
from synthemol.guided_mcts.mcts_utils import calculate_property_score
from synthemol.guided_mcts.policy_network import *
from synthemol.reactions.reaction import Reaction
from synthemol.reactions.real import REAL_REACTIONS
from datetime import datetime


def ucb_score(parent, child, c = 20):
    U = c * child.P * math.sqrt(parent.N) / (child.N + 1)
    return child.Q() + U


class State_Node:
    def __init__(self, molecule, n_building_blocks = 0, prior = 0):
        self.state = molecule                       # current molecule
        self.N = 0                                  # visit count
        self.P = prior                              # prior probability (get from policy network)
        self.W = 0                                  # value sum
        self.children = dict()                      # child node dict(), self.children[a] = child_node
        self.n_building_blocks = n_building_blocks  # number of building blocks involved
        self.reaction_ids = list()                  # list of reactions undergo
        self.property_score = 0                     # property score, nonzero for two-tail lipids
        self.lipid_score = 0
        self.ionizability_score = 0
        self.new_building_block = ''                # new building block added to form the current molecule (action conducted)                
    
    def Q(self):
        if self.N == 0:
            return 0
        return self.W / self.N



class MCTS:
    def __init__(self, policynetwork, max_child_num, smiles_fp_dict, smiles_to_id_dict, root_state = ''):
        self.policynetwork = policynetwork
        self.smiles_fp_dict = smiles_fp_dict
        self.smiles_to_id_dict = smiles_to_id_dict
        self.root_node = State_Node(root_state)
        self.generation = list()
        self.reactions = REAL_REACTIONS
        self.max_child_num = max_child_num
        # with open('all_building_blocks_head.json', 'r') as file:
        #     self.head_search_space = sorted(json.load(file))
        with open('test_head_search_space.json', 'r') as file:
            self.head_search_space = sorted(json.load(file))
        self.head_search_space_split = dict()

    

    def set_head_search_space_split(self, n_play):
        # Shuffle the list to ensure random distribution
        random.shuffle(self.head_search_space)
        
        # Calculate total items to be used for splits
        total_items = n_play * self.max_child_num
        
        # Check if there are enough items to split into the requested sets
        if len(self.head_search_space) < total_items:
            raise ValueError("Not enough items to create the requested number of splits.")
        
        # Create the splits and store them in a dictionary
        for i in range(n_play):
            start_index = i * self.max_child_num
            end_index = start_index + self.max_child_num
            self.head_search_space_split[i] = self.head_search_space[start_index:end_index]


    
    def run(self, n_simulation, save_path, save_path_cleaned, play_j = None):
        # Initialize head search space for the root node
        self.expand(self.root_node, play_j)

        # Run tree search for n_simulation times
        for i in tqdm(range(n_simulation), desc="Simulations"):
            leaf_node, search_path = self.select(simulation_count = i)

            # Evaluate if the leaf node already represents a two-tail lipid
            if leaf_node.n_building_blocks == 3:
                if leaf_node.N > 0:
                    v = leaf_node.property_score
                else:
                    lipid_score, ionizability_score = calculate_property_score(leaf_node.state)
                    v = lipid_score + ionizability_score
                    leaf_node.property_score = v
                    leaf_node.lipid_score = lipid_score
                    leaf_node.ionizability_score = ionizability_score
                
                self.generation.append(search_path)

            # Expand and evaluate (by rollout)
            else:   
                self.expand(leaf_node)   
                lipid_score, ionizability_score = self.rollout(leaf_node)              
                v = lipid_score + ionizability_score
            
            # Backpropagate to update action value and visit count for the chosen path
            self.backpropagate(v, search_path)
        
        # Write generation to log file
        self.save_log_file(save_path, save_path_cleaned)

        # Return search data for training policy network
        search_data = self.record_search_data(self.root_node)

        return search_data, self.smiles_fp_dict
    
    

    def test(self, n_simulation, save_path, save_path_cleaned):

        with open('test_head_search_space.json', 'r') as file:
            self.head_search_space = sorted(json.load(file))
        
        for i in tqdm(range(n_simulation), desc="Simulations"):
            node = self.root_node
            search_path = []

            # Flag indicating whether this path terminate before it generates a two-tail lipid
            flag = False

            while node.n_building_blocks < 3:
                if not node.children:
                    self.expand(node)
                # If no action space
                if not node.children:
                    flag = True
                    break
                
                selected_action = max(node.children, key=lambda a: ucb_score(node, node.children[a]))
                node = node.children[selected_action]

                # Get the current product molecule if first time visited
                if node.N == 0:
                    reaction_id, molecule = self.product(node.state, node.new_building_block)
                    node.state = molecule
                    # If the leaf node is not the head molecule
                    if reaction_id:    
                        node.reaction_ids.append(reaction_id)
                
                search_path.append(node)
            
            # If terminate before a valid generation
            if flag:
                v = 0
            else:
            # Deal with the generated product
                if node.N > 0:
                    v = node.property_score
                else:
                    lipid_score, ionizability_score = calculate_property_score(node.state)
                    v = lipid_score + ionizability_score
                    node.property_score = v
                    node.lipid_score = lipid_score
                    node.ionizability_score = ionizability_score
            
            self.backpropagate(v, search_path)
            
            if not flag:
                self.generation.append(search_path)
        
        # Write generation to log file
        self.save_log_file(save_path, save_path_cleaned)   

        return

    

    def select(self, simulation_count):
        search_path = []
        node = self.root_node

        # Find a leaf node via ucb selection, exploring start
        while node.children:
            # Exploring  start
            if simulation_count < self.max_child_num:
                selected_action = random.choice(list(node.children.keys()))
            else:
                selected_action = max(node.children, key=lambda a: ucb_score(node, node.children[a]))
            node = node.children[selected_action]
            search_path.append(node)
            
        # Get the current product molecule if first time visited
        if node.N == 0:
            reaction_id, molecule = self.product(node.state, node.new_building_block)
            node.state = molecule
            # If the leaf node is not the head molecule
            if reaction_id:    
                node.reaction_ids.append(reaction_id)

        return node, search_path
    

    def expand(self, node, play_j = None):
        action_space = self.next_building_block(node, play_j)
        if not action_space:
            return
        
        prior_probabilities = self.compute_prior(node, action_space)
        n_building_blocks = node.n_building_blocks + 1
        prev_reactions = node.reaction_ids

        # Construct new node for each child
        for i in range(len(action_space)):
            a = action_space[i]
            prior = prior_probabilities[i]
            node.children[a] = State_Node(node.state, n_building_blocks, prior)
            node.children[a].new_building_block = a
            node.children[a].reaction_ids = prev_reactions  
        return

    
    def rollout(self, node):
        while node.n_building_blocks < 3:
            if node.children:
                action_space = list(node.children.keys())
            else:
                action_space = self.next_building_block(node)
            # Suppose the given node can no longer react
            if not action_space:
                return 0, 0
            # Rollout by performing random actions and evaluate the generated final product
            a = random.choice(action_space)
            _, molecule = self.product(node.state, a)
            node = State_Node(molecule, node.n_building_blocks + 1)

        lipid_score, ionizability_score = calculate_property_score(node.state)

        return lipid_score, ionizability_score


    def backpropagate(self, v, search_path):
        for node in search_path:
            node.N += 1
            node.W += v
            self.root_node.N += 1
        return


    def compute_prior(self, node, action_space):
        if node.n_building_blocks == 1:
            state_fp = self.smiles_fp_dict[node.state]
        else:
            state_fp = smiles_to_fp(node.state)
            # Update state_fp_dict with new states (intermediate product)
            self.smiles_fp_dict[node.state] = state_fp

        input_tensors = [torch.cat((state_fp, self.smiles_fp_dict[action]), dim=0) for action in action_space]
        input_batch = torch.stack(input_tensors)
        logits = self.policynetwork(input_batch)
        prior_probabilities = F.softmax(logits, dim=0)

        return prior_probabilities.detach().numpy()


    # Run reaction over the provided two molecules, return a product
    def product(self, mol1, mol2):
        # Deal with root state
        if not mol1:
            return (None, mol2)

        molecules = (mol1, mol2)
        matching_reactions = self.get_reactions_for_molecules(molecules)
        results = []
        for reaction, molecule_to_reactant_index in matching_reactions:
            # Put molecules in the right order for the reaction
            molecules = sorted(molecules, key=lambda frag: molecule_to_reactant_index[frag])

            # Run reaction
            products = reaction.run_reactants(molecules)
            products = [Chem.MolToSmiles(Chem.RemoveHs(product[0])) for product in products]
            products = set(products)
            for product in products:
                results.append((reaction.id, product))
        
        selected_product = random.choice(results)
        return selected_product
    

    def get_reactions_for_molecules(self, molecules):
        """Get all reactions that can be run on the given molecules.

        :param molecules: A tuple of SMILES strings representing the molecules to run reactions on.
        :return: A list of tuples, where each tuple contains a reaction and a dictionary mapping
                 the molecules to the indices of the reactants they match.
        """
        matching_reactions = []

        # Check each reaction to see if it can be run on the given molecules
        for reaction in self.reactions:
            # Skip reaction if the number of molecules doesn't match the number of reactants
            if len(molecules) != reaction.num_reactants:
                continue

            # For each molecule, get a list of indices of reactants it matches
            reactant_matches_per_molecule = [
                reaction.get_reactant_matches(smiles=molecule)
                for molecule in molecules
            ]

            # Include every assignment of molecules to reactants that fills all the reactants
            for matched_reactant_indices in itertools.product(*reactant_matches_per_molecule):
                if len(set(matched_reactant_indices)) == reaction.num_reactants:
                    molecule_to_reactant_index = dict(zip(molecules, matched_reactant_indices))
                    matching_reactions.append((reaction, molecule_to_reactant_index))

        return matching_reactions


    # Get action space of the current node, return list of available next building blocks
    def next_building_block(self, node, play_j = None):
        sample_random = random.Random()
        sample_random.seed(442)
        # Return a subset of the head search space if root node
        if not node.state:
            if play_j:
                return self.head_search_space_split[play_j]
            else:
                return sample_random.sample(self.head_search_space, self.max_child_num)

        molecule = node.state
        reactive_space = []
        
        # Loop through each reaction
        for reaction in self.reactions:
            # Get indices of the reactants in this reaction
            reactant_indices = set(range(reaction.num_reactants))
            # For given molecule, get a list of indices of reactants it matches
            reactant_matches_per_molecule = [reaction.get_reactant_matches(smiles=molecule)]
            for matched_reactant_indices in itertools.product(*reactant_matches_per_molecule):
                matched_reactant_indices = set(matched_reactant_indices)
                for index in sorted(reactant_indices - matched_reactant_indices):
                    reactive_space += reaction.reactants[index].allowed_building_blocks

        # Remove duplicates but maintain order for reproducibility
        reactive_space = list(dict.fromkeys(reactive_space))
        if len(reactive_space) > self.max_child_num:
            action_space = sample_random.sample(reactive_space, self.max_child_num)
            return action_space
        else:
            return reactive_space


    # Write generated products into log files
    def save_log_file(self, save_path, save_path_cleaned):
        log_data = {
            'final_product': [search_path[2].state for search_path in self.generation],
            'property_score': [search_path[2].property_score for search_path in self.generation],
            'lipid-like': [search_path[2].lipid_score for search_path in self.generation],
            'ionizability': [search_path[2].ionizability_score for search_path in self.generation],
            'lipid_head': [search_path[0].state for search_path in self.generation],
            'lipid_head_id': [self.smiles_to_id_dict[search_path[0].state] for search_path in self.generation],
            'reaction_1_id': [search_path[2].reaction_ids[0] for search_path in self.generation],
            'lipid_tail_1': [search_path[1].new_building_block for search_path in self.generation],
            'lipid_tail_1_id': [self.smiles_to_id_dict[search_path[1].new_building_block] for search_path in self.generation],
            'intermediate_product': [search_path[1].state for search_path in self.generation],
            'reaction_2_id': [search_path[2].reaction_ids[1] for search_path in self.generation],
            'lipid_tail_2': [search_path[2].new_building_block for search_path in self.generation],
            'lipid_tail_2_id': [self.smiles_to_id_dict[search_path[2].new_building_block] for search_path in self.generation],
        }

        df = pd.DataFrame(log_data)
        df.to_csv(save_path, index=False)

        df_clean = df.drop_duplicates()
        df_clean.to_csv(save_path_cleaned, index=False) 
        print('Number of unique generated two-tail lipids:', len(df_clean))

        return


    # Return search counts for all state-action pairs for training policy network
    def record_search_data(self, node, search_data=None):
        # Deal with root node, initialize the search data dict
        if not search_data:
            search_data = dict()

        if node.N > 1:
            # Iterate over each child of the current node
            for action, child in node.children.items():
                # Use the state-action pair ID as key and child's N as value
                search_data[(node.state, action)] = (child.N, child.n_building_blocks)
                # Recursively process each child
                self.record_search_data(child, search_data)

        return search_data