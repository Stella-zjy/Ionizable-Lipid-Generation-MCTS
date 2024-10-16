import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import random
import json
import itertools
import ast
import sys
import os

# Adds the project root directory to sys.path
current_dir = os.path.dirname(__file__)  # Directory of the script
parent_dir = os.path.dirname(current_dir)  # Parent directory ('some_subfolder')
grandparent_dir = os.path.dirname(parent_dir)  # Grandparent directory ('synthemol')

# Add the grandparent directory (above 'synthemol') to sys.path
sys.path.insert(0, os.path.abspath(grandparent_dir))

from synthemol.guided_mcts.mcts_utils import *
from synthemol.guided_mcts.mcts import *
from synthemol.guided_mcts.policy_network import *

from synthemol.reactions.reaction import Reaction
from synthemol.reactions.real import REAL_REACTIONS
from synthemol.reactions.utils import load_and_set_allowed_reaction_building_blocks, set_all_building_blocks


def setup_reaction_tail_building_blocks(reactions, tail_building_blocks, reaction_to_building_blocks_path):
    set_all_building_blocks(reactions, tail_building_blocks)
    load_and_set_allowed_reaction_building_blocks(reactions, reaction_to_building_blocks_path)
    return


# Convert the comma-separated strings back to tensors
def string_to_tensor(fp_string):
    fp_list = list(map(float, fp_string.split(',')))  # Convert string to list of floats
    return torch.tensor(fp_list, dtype=torch.float32)


def main():
    reaction_to_building_blocks_path = 'data/Data/RawLipid/reaction_to_tail_building_blocks_guided_mcts.pkl'
    tail_building_blocks = pd.read_csv('data/Data/RawLipid/purchasable_tails_unique.csv')['smiles'].tolist()
    
    print('Loading and setting allowed building blocks for each reaction...')
    setup_reaction_tail_building_blocks(REAL_REACTIONS, tail_building_blocks, reaction_to_building_blocks_path)


    # Load precomputed fingerprint for all building blocks
    building_block_data = pd.read_csv('data/Data/RawLipid/lipid_building_blocks_guided_mcts.csv')
    # Conver fingerprint to tensor format
    building_block_data['fp_tensor'] = building_block_data['fp'].apply(string_to_tensor)
    # Create a dictionary with SMILES as keys and tensor fingerprints as values
    smiles_fp_dict = dict(zip(
        building_block_data['smiles'],
        building_block_data['fp_tensor']
    ))

    # Map building blocks SMILES to IDs, IDs to SMILESs
    smiles_to_id_dict = dict(zip(
        building_block_data['smiles'],
        building_block_data['ID']
    ))
    id_to_smiles_dict = dict(zip(
        building_block_data['ID'],
        building_block_data['smiles']
    ))
    
    # Initialize policy network
    policynetwork = PolicyNetwork()

    n_iterations = 10
    for i in range(n_iterations):
        print('Iteration', i)
        dir_path = f'data/Data/guided_mcts_generation/iteration_{i}'
        os.makedirs(dir_path, exist_ok=True)

        # load updated policy network
        if i > 0:
            policynetwork = PolicyNetwork()
            policynetwork.load_state_dict(torch.load(model_save_path))

        policynetwork.eval()
        
        # Initialize MCTS
        max_child_num = 200
        mcts = MCTS(policynetwork, max_child_num, smiles_fp_dict, smiles_to_id_dict)


        # MCTS simulations for collecting search data
        n_play = 10
        n_simulation = 10000
        print('Start Monte Carlo Tree Search')
        total_search_data = dict()
        total_updated_smiles_fp_dict = dict()
        total_search_data_store = dict()

        # Initialize different sets of head molecules for each play
        mcts.set_head_search_space_split(n_play)
        
        for j in range(n_play): 
            print('Play', j)
            save_path = dir_path + f'/molecules_play_{j}.csv'
            save_path_cleaned = dir_path + f'/cleaned_molecules_play_{j}.csv'
            start_time = datetime.now()
            
            search_data, updated_smiles_fp_dict = mcts.run(n_simulation, save_path, save_path_cleaned, j)

            total_search_data[j] = search_data
            total_updated_smiles_fp_dict[j] = updated_smiles_fp_dict

            # Convert current search_data for serialization
            serializable_search_data = {str(key): list(value) for key, value in search_data.items()}
            total_search_data_store[j] = serializable_search_data
            
            print(f'MCTS Runtime for Play {j}:', datetime.now() - start_time)
        
        print(f'Store search data of iteration {i} to a local json file...')
        # Store total search data to a local json file
        search_data_save_path = f'data/Data/guided_mcts_generation/search_data_iter_{i}.json'
        with open(search_data_save_path, 'w') as json_file:
            json.dump(total_search_data_store, json_file)

        print('Number of state-action pairs recorded:', sum(len(total_search_data[j]) for j in range(n_play)))

        # Train policy network
        model_save_path = f'data/Models/guided_mcts_policy_network/policy_network_iter_{i}.pth'
        epochs = 20
        train_start_time = datetime.now()
        train_policynetwork(policynetwork, model_save_path, epochs, total_search_data, total_updated_smiles_fp_dict)
        print('Training Time:', datetime.now() - train_start_time)


   
if __name__ == "__main__":
    main()
    # search_data_save_path = 'data/Data/guided_mcts_generation/search_data_iter_0_test.json'
    # with open(search_data_save_path, 'r') as json_file:
    #     loaded_data = json.load(json_file)
    #     # Convert the string keys back to tuples and tuple values
    #     original_data = {tuple(eval(key)): tuple(value) for key, value in loaded_data.items()}

    # values = [value for value in original_data.values()]
    # value_counts = Counter(values)
    # print(value_counts)
