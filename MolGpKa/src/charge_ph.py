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

# from utils.ionization_group import get_ionization_aid
# from utils.descriptor import mol2vec
# from utils.net import GCNNet
from MolGpKa.src.utils.ionization_group import get_ionization_aid
from MolGpKa.src.utils.descriptor import mol2vec
from MolGpKa.src.utils.net import GCNNet

import matplotlib.pyplot as plt


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


# Conduct ionizability binary classification for the given single molecule, output 1 for ionizable and 0 for non-ionizable
def ionizability_classifier_single_molecule(smiles, model_acid, model_base):
    mol = Chem.MolFromSmiles(smiles)
    if calculate_net_charge(mol, model_acid, model_base, 5) < 0.1:
        return 0
    if calculate_net_charge(mol, model_acid, model_base, 7.4) < 0:
        return 0
    return 1
