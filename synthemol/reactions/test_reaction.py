from reaction import Reaction
from real import REAL_REACTIONS
from query_mol import QueryMol
import pandas as pd
import json


building_block_heads = pd.read_csv('/scratch/jz610/GitHub/SyntheMol-Lipid/data/Data/RawLipid/heads.csv')['smiles']
reactions = REAL_REACTIONS

output = []
for building_block in building_block_heads:
    if any(reactant.has_match(building_block) for reaction in reactions for reactant in reaction.reactants):
        output.append(building_block)         

print(len(output))

# Saving the list to a file using JSON
with open('all_building_blocks_head.json', 'w') as file:
    json.dump(output, file)
