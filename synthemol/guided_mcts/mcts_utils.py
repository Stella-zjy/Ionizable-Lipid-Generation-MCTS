import math
import torch
import numpy as np
import pandas as pd

from chemfunc import compute_fingerprint
from synthemol.models import (
    chemprop_load,
    chemprop_load_scaler,
    chemprop_predict_on_molecule
)
# from synthemol.guided_mcts.mcts import *
from MolGpKa.src.charge_ph import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_base = load_model("MolGpKa/models/weight_base.pth", device)
model_acid = load_model("MolGpKa/models/weight_acid.pth", device)

lipid_classifier_model_path = 'data/Models/lipid_classifier_chemprop_1_epochs/model_0.pt'
lipid_classifier_model = chemprop_load(model_path=lipid_classifier_model_path)
lipid_classifier_scaler = chemprop_load_scaler(model_path=lipid_classifier_model_path)

# Evaluate the given molecule (lipid-like and ionizable)
def calculate_property_score(smiles):

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    fingerprint = compute_fingerprint(smiles, fingerprint_type='morgan')
    lipid_score = chemprop_predict_on_molecule(lipid_classifier_model, smiles, fingerprint, lipid_classifier_scaler)

    ionizability_score = ionizability_classifier_single_molecule(smiles, model_acid, model_base, device)

    return lipid_score, ionizability_score