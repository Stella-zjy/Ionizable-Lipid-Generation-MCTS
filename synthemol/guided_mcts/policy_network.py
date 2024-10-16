import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime


class PolicyNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 1024),  # Reduce dimensionality
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)       # Output layer for regression
        )

    def forward(self, x):
        x = self.layers(x)        # Pass input through the network
        x = x.squeeze(1)
        return x      


# Custom Loss Function
def custom_loss(outputs, targets, c, tau = 2):
    loss = 0
    N = len(outputs)
    count = c * N
    for _ in range(count):
        i, j = random.sample(range(N), 2)
        if targets[i][1] != targets[j][1]:
            continue
        pred = outputs[i] - outputs[j]
        targ = 1/tau * (np.log(targets[i][0]) - np.log(targets[j][0]))
        loss += abs(pred - targ)
    return loss / count



def smiles_to_fp(smiles, radius=2, nBits=1024):
    """Convert a SMILES string to a Morgan fingerprint as a PyTorch tensor."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle cases where the molecule couldn't be parsed
        return torch.zeros((nBits,), dtype=torch.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return torch.tensor(list(fp), dtype=torch.float32)



def dataset_construction_level_seperate(total_search_data, total_updated_smiles_fp_dict):
    # Map to store each unique state with a unique ID
    state_to_id = {}
    current_id = 0

    def select_sort_and_stack(data, N):
        state_ids = list(set(x[0] for x in data))
        selected_state_ids = random.sample(state_ids, min(N, len(state_ids)))
        
        filtered_data = [x for x in data if x[0] in selected_state_ids]
        filtered_data.sort(key=lambda x: x[0])
        
        features = torch.stack([x[1] for x in filtered_data])
        targets = torch.stack([x[2] for x in filtered_data])
        
        return features, targets
    
    all_features_1 = []
    all_features_2 = []
    all_features_3 = []
    all_targets_1 = []
    all_targets_2 = []
    all_targets_3 = []

    for i in range(len(total_search_data)):
        search_data = total_search_data[i]
        updated_smiles_fp_dict = total_updated_smiles_fp_dict[i]

        data_1 = []
        data_2 = []
        data_3 = []
        
        # Assign IDs to unique states and prepare tensors
        for (state, action), (N, n_building_blocks) in search_data.items():
            # Assign a unique ID to each state
            if state not in state_to_id:
                # Separate the root state for each tree search
                if state == '' and str(i) not in state_to_id:
                    state_to_id[str(i)] = current_id
                    current_id += 1
                elif state == '' and str(i) in state_to_id:
                    pass
                else:
                    state_to_id[state] = current_id
                    current_id += 1
            
            # Retrieve the fingerprint data for state and action
            state_fp = updated_smiles_fp_dict[state]
            action_fp = updated_smiles_fp_dict[action]

            # Concatenate the fingerprints to form the input feature
            concatenated_fp = torch.cat((state_fp, action_fp))

            # Prepare the target tensor with target N + 1
            if state == '':
                state_id = state_to_id[str(i)]
            else:
                state_id = state_to_id[state]
            target = torch.tensor([N + 1, state_id])

            # Create tuple and append to the corresponding list
            data_tuple = (state_id, concatenated_fp, target)
            if n_building_blocks == 1:
                data_1.append(data_tuple)
            elif n_building_blocks == 2:
                data_2.append(data_tuple)
            else:
                data_3.append(data_tuple)

        features_1, targets_1 = select_sort_and_stack(data_1, 1)
        features_2, targets_2 = select_sort_and_stack(data_2, 10)
        features_3, targets_3 = select_sort_and_stack(data_3, 100)
        all_features_1.append(features_1)
        all_features_2.append(features_2)
        all_features_3.append(features_3)
        all_targets_1.append(targets_1)
        all_targets_2.append(targets_2)
        all_targets_3.append(targets_3)
    
    final_features_1 = torch.cat(all_features_1, dim=0)
    final_targets_1 = torch.cat(all_targets_1, dim=0)
    final_features_2 = torch.cat(all_features_2, dim=0)
    final_targets_2 = torch.cat(all_targets_2, dim=0)
    final_features_3 = torch.cat(all_features_3, dim=0)
    final_targets_3 = torch.cat(all_targets_3, dim=0)
        

    return final_features_1, final_targets_1, final_features_2, final_targets_2, final_features_3, final_targets_3



def train_policynetwork(model, model_save_path, epochs, total_search_data, total_updated_smiles_fp_dict):
    # Set up training dataset and optimiser
    print('Constructing training dataset...')
    features_1, targets_1, features_2, targets_2, features_3, targets_3 = dataset_construction_level_seperate(total_search_data, total_updated_smiles_fp_dict)
    dataset_1 = TensorDataset(features_1, targets_1)
    dataset_2 = TensorDataset(features_2, targets_2)
    dataset_3 = TensorDataset(features_3, targets_3)

    batch_size = 200
    dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, shuffle=False)
    dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, shuffle=False)
    dataloader_3 = DataLoader(dataset_3, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')

    # Training Loop
    print('Model training...')
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_targets in dataloader_1:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = custom_loss(outputs, batch_targets, 100)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        for batch_features, batch_targets in dataloader_2:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = custom_loss(outputs, batch_targets, 10)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        for batch_features, batch_targets in dataloader_3:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = custom_loss(outputs, batch_targets, 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / (len(dataloader_1)+len(dataloader_2)+len(dataloader_3))
        print(f'Epoch {epoch+1}, Loss: {average_loss}')

        # Check if the current model is the best one; if so, save it
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with loss: {best_loss}')

    print('Training completed. Best model state saved to', model_save_path)



# if __name__ == "__main__":
#     model = PolicyNetwork()
#     model.eval()
