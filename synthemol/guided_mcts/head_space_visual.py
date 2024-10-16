from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import json
import random


# Convert the comma-separated strings back to tensors
def string_to_tensor(fp_string):
    fp_list = list(map(float, fp_string.split(',')))  # Convert string to list of floats
    return torch.tensor(fp_list, dtype=torch.float32)


def smiles_to_fp(smiles, radius=2, nBits=1024):
    """Convert a SMILES string to a Morgan fingerprint as a PyTorch tensor."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle cases where the molecule couldn't be parsed
        return ",".join(["0.0"] * nBits)  # Return zero-filled string
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return ",".join(map(str, list(fp)))  # Convert bits to string separated by commas



# Load precomputed fingerprint for all building blocks
building_block_data = pd.read_csv('data/Data/RawLipid/lipid_building_blocks_guided_mcts.csv')
# Conver fingerprint to tensor format
building_block_data['fp_tensor'] = building_block_data['fp'].apply(string_to_tensor)

head_space_fp = building_block_data['fp_tensor'][:12338]
train_fp_numpy = np.stack(head_space_fp.apply(lambda x: x.numpy()))



# Test set head building blocks
with open('all_heads.json', 'r') as file:
    all_heads = json.load(file)

with open('all_building_blocks_head.json', 'r') as file:
    train_head_search_space = json.load(file)

head_subset = random.sample(all_heads, 300)
test_head_search_space = []
count = 0
for i in range(300):
    if head_subset[i] not in train_head_search_space:
        test_head_search_space.append(head_subset[i])
        count += 1
        if count == 200:
            break

# Create a DataFrame
df = pd.DataFrame({
    'smiles': test_head_search_space
})

df['fp'] = df['smiles'].apply(smiles_to_fp) 
df.to_csv('data/Data/RawLipid/test_head_search_space.csv')

df['fp_tensor'] = df['fp'].apply(string_to_tensor)
test_fp_numpy = np.stack(df['fp_tensor'].apply(lambda x: x.numpy()))



# Combine both arrays for t-SNE application
combined_fp = np.vstack([train_fp_numpy, test_fp_numpy])

# Apply t-SNE to the combined dataset
tsne = TSNE(n_components=2, random_state=42)
combined_fp_reduced = tsne.fit_transform(combined_fp)

# Extract the t-SNE results for each dataset
train_tsne = combined_fp_reduced[:len(train_fp_numpy)]
test_tsne = combined_fp_reduced[len(train_fp_numpy):]

# Plotting the results
plt.figure(figsize=(10, 8))
plt.scatter(train_tsne[:, 0], train_tsne[:, 1], alpha=0.5, color='gray', label='Train Data', s=12)
plt.scatter(test_tsne[:, 0], test_tsne[:, 1], alpha=0.5, color='red', label='Test Data', s=12)
plt.title('t-SNE Visualization of Lipid Head Building Blocks', fontsize = 18)
plt.xlabel('t-SNE Component 1', fontsize = 16)
plt.ylabel('t-SNE Component 2', fontsize = 16)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.legend(fontsize=14)
# plt.colorbar(label='Density')
plt.savefig('synthemol/guided_mcts/train_and_test_head_space_visual_2.png')


