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
    building_block_data = pd.read_csv('data/Data/RawLipid/test_lipid_building_blocks_guided_mcts.csv')
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


    max_child_num = 200
    n_simulation = 10000


    # Test for random initialized policy network
    save_path = 'data/Data/guided_mcts_generation_test/molecules_iter_0.csv'
    save_path_cleaned = 'data/Data/guided_mcts_generation_test/cleaned_molecules_iter_0.csv'
    policynetwork = PolicyNetwork()
    policynetwork.eval()
    
    # Initialize MCTS
    start_time = datetime.now()
    mcts = MCTS(policynetwork, max_child_num, smiles_fp_dict, smiles_to_id_dict)
    _, _ = mcts.run(n_simulation, save_path, save_path_cleaned)
    print('(Iteration 0) MCTS Simulation Time:', datetime.now() - start_time)

    model_save_dir = 'data/Models/guided_mcts_policy_network'
    n_iterations = 10

    for i in range(n_iterations):
        start_time = datetime.now()
        model_save_path = model_save_dir + f'/policy_network_iter_{i}.pth'

        # Load model
        policynetwork = PolicyNetwork()
        policynetwork.load_state_dict(torch.load(model_save_path))
        policynetwork.eval()

        save_path = f'data/Data/guided_mcts_generation_test/molecules_iter_{i+1}.csv'
        save_path_cleaned = f'data/Data/guided_mcts_generation_test/cleaned_molecules_iter_{i+1}.csv'

        mcts = MCTS(policynetwork, max_child_num, smiles_fp_dict, smiles_to_id_dict)
        _, _ = mcts.run(n_simulation, save_path, save_path_cleaned)
        print(f'(Iterarion {i+1}) MCTS Simulation Time:', datetime.now() - start_time)


   
if __name__ == "__main__":
    main()