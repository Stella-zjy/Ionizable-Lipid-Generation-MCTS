#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize

import os.path as osp
import numpy as np
import pandas as pd

import torch
from utils.ionization_group import get_ionization_aid
from utils.descriptor import mol2vec
from utils.net import GCNNet

import matplotlib.pyplot as plt
import random
import time

root = osp.abspath(osp.dirname(__file__))

def load_model(model_file, device="cpu"):
    model= GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model

def model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka

def predict_acid(mol, model_acid):
    acid_idxs= get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid)
        acid_res.update({aid:apka})
    return acid_res

def predict_base(mol, model_base):
    base_idxs= get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base) 
        base_res.update({aid:bpka})  
    return base_res

def predict(mol, model_acid, model_base, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol, model_base)
    acid_dict = predict_acid(mol, model_acid)
    return base_dict, acid_dict

# def predict_for_protonate(mol, uncharged=True):
#     if uncharged:
#         un = rdMolStandardize.Uncharger()
#         mol = un.uncharge(mol)
#         mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
#     mol = AllChem.AddHs(mol)
#     base_dict = predict_base(mol)
#     acid_dict = predict_acid(mol)
#     return base_dict, acid_dict, mol


# Calculate the net charge of a molecule udner a given pH value
def calculate_net_charge(mol, model_acid, model_base, ph):
    base_dict, acid_dict = predict(mol, model_acid, model_base)
    net_charge = 0.0
    for b in base_dict.keys():
        pka = base_dict[b]
        fractional_charge = (10 ** (pka - ph))/(1 + 10 ** (pka - ph))
        net_charge += fractional_charge
    for a in acid_dict.keys():
        pka = acid_dict[a]
        fractional_charge = (10 ** (ph - pka))/(1 + 10 ** (ph - pka))
        net_charge -= fractional_charge
    return net_charge

# Plot the charge-pH plot of a molecule, plot accuracy is customed by given step size
def charge_ph_plot(mol, model_acid, model_base, fig_path, step_size = 1):
    phs = np.arange(0, 7.1, step_size).tolist() + [7.4] +  np.arange(8, 14.1, step_size).tolist()
    charges = [calculate_net_charge(mol, model_acid, model_base, ph) for ph in phs]
    plt.figure()
    plt.plot(phs, charges, label='Charge vs pH')
    plt.xlabel('pH')
    plt.ylabel('charge')
    plt.title('charge-pH plot')
    plt.grid(True)
    plt.axvline(x=7.4, color='r', linestyle='--', label='pH=7.4')
    plt.axhline(y=0, color='grey', linestyle='-', label='Charge=0')
    plt.legend()
    plt.savefig(fig_path)


# Conduct ionizability filtering for all SMILES strings in the input file, results can be written into an output file
def ionizability_filter(input_file_path, output_file_path, model_acid, model_base, save_file = True):
    total_rows = 2752482
    n_samples = 1000
    skip_rows = set(random.sample(range(1, total_rows), total_rows - n_samples)) 

    df = pd.read_csv(input_file_path, skiprows=lambda x: x in skip_rows)

    start_time = time.time()

    sample_smiles = df['smiles'].tolist()
    
    filtered_list = []
    non_selected_list = []

    for smi in sample_smiles:
        mol = Chem.MolFromSmiles(smi)
        if calculate_net_charge(mol, model_acid, model_base, 5) < 0.1:
            non_selected_list.append(smi)
            continue
        if calculate_net_charge(mol, model_acid, model_base, 7.4) < 0:
            non_selected_list.append(smi)
            continue
        filtered_list.append(smi)

    end_time = time.time()

    print(len(filtered_list))
    print('Runtime', end_time - start_time)

    if save_file:
        df_out = pd.DataFrame(filtered_list, columns=['smiles'])
        df_out.to_csv(output_file_path, index=False)

    return filtered_list, non_selected_list


# Conduct ionizability binary classification for the given single molecule, output 1 for ionizable and 0 for non-ionizable
def ionizability_classifier_single_molecule(smiles, model_acid, model_base):
    mol = Chem.MolFromSmiles(smiles)
    if calculate_net_charge(mol, model_acid, model_base, 5) < 0.1:
        return 0
    if calculate_net_charge(mol, model_acid, model_base, 7.4) < 0:
        return 0
    return 1



if __name__ == "__main__":
    # ==================== Ionizable Head Filtering  ====================
    input_file_path = '/scratch/jz610/GitHub/SyntheMol-Lipid/data/Data/RawLipid/heads.csv'
    output_file_path = '/scratch/jz610/GitHub/SyntheMol-Lipid/data/Data/RawLipid/charge_filtered_heads.csv'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base = load_model(osp.join(root, "../models/weight_base.pth"), device)
    model_acid = load_model(osp.join(root, "../models/weight_acid.pth"), device)

    filtered_list, non_selected_list = ionizability_filter(input_file_path, output_file_path, model_acid, model_base, False)

    # n = 10
    # selected_sample = random.sample(filtered_list, n)
    # non_selected_sample = random.sample(non_selected_list, n)

    # print('Ionizable molecules')
    # for i in range(n):
    #     print(i, selected_sample[i])
    #     fig_path = f'../charge_pH_plots/ionizable_sample_{i}.png'
    #     mol = Chem.MolFromSmiles(selected_sample[i])
    #     charge_ph_plot(mol, model_acid, model_base, fig_path)
    
    # print('\n')
    # print('Non-Ionizable molecules')
    # for i in range(n):
    #     print(i, non_selected_sample[i])
    #     fig_path = f'../charge_pH_plots/non_ionizable_sample_{i}.png'
    #     mol = Chem.MolFromSmiles(non_selected_sample[i])
    #     charge_ph_plot(mol, model_acid, model_base, fig_path)
