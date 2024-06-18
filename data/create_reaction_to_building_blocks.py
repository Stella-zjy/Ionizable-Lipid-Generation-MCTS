import pickle
import pandas as pd

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from synthemol.reactions.reaction import Reaction
from synthemol.reactions.real import REAL_REACTIONS

import time

reactions = REAL_REACTIONS
reaction_to_reactant_to_building_blocks = dict()

# List of all building blocks (both heads and tails SMILES)
building_blocks_list = pd.read_csv('data/Data/RawLipid/lipid_building_blocks_try.csv')['smiles'].tolist()

start_time = time.time()
# Iterate over all reactions
for reaction in reactions:
    print(reaction)
    reaction_to_reactant_to_building_blocks[reaction.id] = dict()
    # Iterate over all reactants of each reaction
    for reactant_index, reactant in enumerate(reaction.reactants):
        reaction_to_reactant_to_building_blocks[reaction.id][reactant_index] = list()
        for building_block in building_blocks_list:
            if reactant.has_substruct_match(building_block):
                reaction_to_reactant_to_building_blocks[reaction.id][reactant_index].append(building_block)

print(time.time() - start_time)

# File path where you want to save the pickle file
file_path = 'data/Data/RawLipid/reaction_to_lipid_building_blocks_try.pkl'

# Open a file for writing the pickle data
with open(file_path, 'wb') as file:
    # Use pickle.dump to serialize the dictionary and write it to the file
    pickle.dump(reaction_to_reactant_to_building_blocks, file)


# Test PKL file
with open(file_path, 'rb') as f:
        reaction_to_reactant_to_building_blocks: dict[int, dict[int, set[str]]] = pickle.load(f)

for reaction in reactions:
    print(reaction)
    for reactant_index, reactant in enumerate(reaction.reactants):
        print(len(reaction_to_reactant_to_building_blocks[reaction.id][reactant_index]))
            