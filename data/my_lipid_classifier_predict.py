"""Make predictions with a model or ensemble of models and save them to a file."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemfunc import compute_fingerprints
from chemprop.data import set_cache_graph, set_cache_mol
from tqdm import tqdm
from chemprop.models import MoleculeModel
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from chemprop.train import predict as _chemprop_predict
from chemprop.utils import load_checkpoint

SMILES_COL = 'smiles'
MODEL_TYPES = 'chemprop'
FINGERPRINT_TYPES = 'morgan'


def chemprop_load(
        model_path: Path,
        device: torch.device = torch.device('cpu')
) -> MoleculeModel:
    """Loads a Chemprop model.

    :param model_path: A path to a Chemprop model.
    :param device: The device on which to load the model.
    :return: A Chemprop model.
    """
    return load_checkpoint(
        path=str(model_path),
        device=device
    ).eval()



def chemprop_predict(
        model: MoleculeModel,
        smiles: list[str],
        fingerprints: np.ndarray | None = None,
        num_workers: int = 0
) -> np.ndarray:
    """Predicts molecular properties using a Chemprop model.

    :param model: A Chemprop model.
    :param smiles: A list of SMILES strings.
    :param fingerprints: A 2D array of molecular fingerprints (num_molecules, num_features).
    :param num_workers: The number of workers for the data loader.
    :return: A 1D array of predicted properties (num_molecules,).
    """
    # Set up data loader
    data_loader = chemprop_build_data_loader(
        smiles=smiles,
        fingerprints=fingerprints,
        num_workers=num_workers
    )

    # Make predictions
    preds = np.array(_chemprop_predict(model=model, data_loader=data_loader))[:, 0]

    return preds

def chemprop_build_data_loader(
        smiles: list[str],
        fingerprints: np.ndarray | None = None,
        properties: list[int] | None = None,
        shuffle: bool = False,
        num_workers: int = 0
) -> MoleculeDataLoader:
    """Builds a chemprop MoleculeDataLoader.

    :param smiles: A list of SMILES strings.
    :param fingerprints: A 2D array of molecular fingerprints (num_molecules, num_features).
    :param properties: A list of molecular properties (num_molecules,).
    :param shuffle: Whether to shuffle the data loader.
    :param num_workers: The number of workers for the data loader.
                        Zero workers needed for deterministic behavior and faster training/testing when CPU only.
    :return: A Chemprop data loader.
    """
    if fingerprints is None:
        fingerprints = [None] * len(smiles)

    if properties is None:
        properties = [None] * len(smiles)
    else:
        properties = [[float(prop)] for prop in properties]

    return MoleculeDataLoader(
        dataset=MoleculeDataset([
            MoleculeDatapoint(
                smiles=[smiles],
                targets=prop,
                features=fingerprint,
            ) for smiles, fingerprint, prop in zip(smiles, fingerprints, properties)
        ]),
        num_workers=num_workers,
        shuffle=shuffle
    )


def predict(
        data_path: Path,
        model_path: Path,
        model_type: MODEL_TYPES,
        save_path: Path | None = None,
        smiles_column: str = SMILES_COL,
        preds_column_prefix: str | None = None,
        fingerprint_type: str = FINGERPRINT_TYPES,
        average_preds: bool = False,
        num_workers: int = 0,
        use_gpu: bool = False,
        no_cache: bool = False
) -> None:
    """Make predictions with a model or ensemble of models and save them to a file.

    :param data_path: Path to a CSV file containing SMILES.
    :param model_path: Path to a directory of model checkpoints or to a specific PKL or PT file containing a trained model.
    :param model_type: Type of model to use.
    :param save_path: Path to a CSV file where model predictions will be saved. If None, defaults to data_path.
    :param smiles_column: Name of the column containing SMILES.
    :param preds_column_prefix: Prefix for the column containing model predictions.
    :param fingerprint_type: Type of fingerprints to use as input features.
    :param average_preds: Whether to average predictions across models for an ensemble model.
    :param num_workers: Number of workers for the data loader (only applicable to chemprop model type).
    :param use_gpu: Whether to use GPU (only applicable to chemprop model type).
    :param no_cache: Whether to disable caching (only applicable to chemprop model type).
                     Turn off caching when making predictions on large datasets
    """
    # Disable Chemprop caching for prediction to avoid memory issues with large datasets
    if no_cache:
        set_cache_graph(False)
        set_cache_mol(False)

    # Load SMILES
    data = pd.read_csv(data_path)
    data_smiles = pd.DataFrame({smiles_column: data.loc[:, smiles_column]})
    smiles = list(data_smiles[smiles_column])

    # Check compatibility of model and fingerprint type
    if model_type != 'chemprop' and fingerprint_type is None:
        raise ValueError('Must define fingerprint_type if using sklearn model.')

    # Compute fingerprints
    if fingerprint_type is not None:
        fingerprints = compute_fingerprints(smiles, fingerprint_type=fingerprint_type)
    else:
        fingerprints = None

    
    # Load models
    if model_type == 'chemprop':
        # Set device
        if use_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Ensure reproducibility
        torch.manual_seed(0)

        if device.type == 'cpu':
            torch.use_deterministic_algorithms(True)

        models = [chemprop_load(model_path=model_path, device=device)]
    
    # Make predictions
    if model_type == 'chemprop':
        preds = np.array([
            chemprop_predict(
                model=model,
                smiles=smiles,
                fingerprints=fingerprints,
                num_workers=num_workers
            ) for model in tqdm(models, desc='models')
        ])
    

    if average_preds:
        preds = np.mean(preds, axis=0)

    # Define model string
    model_string = f'{model_type}{f"_{fingerprint_type}" if fingerprint_type is not None else ""}'
    preds_string = f'{f"{preds_column_prefix}_" if preds_column_prefix is not None else ""}{model_string}'

    if average_preds:
        # data_smiles[f'{preds_string}_ensemble_preds'] = preds
        data[f'{preds_string}_ensemble_preds'] = preds
    else:
        for model_num, model_preds in enumerate(preds):
            # data_smiles[f'{preds_string}_model_{model_num}_preds'] = model_preds
            data[f'{preds_string}_model_{model_num}_preds'] = model_preds

   
    # data_smiles.to_csv(save_path, index=False)
    data.to_csv(save_path, index=False)


if __name__ == '__main__':
    data_path = 'data/Data/RawLipid/lipid_building_blocks.csv'
    model_path = 'data/Models/lipid_classifier_chemprop_1_epochs/model_0.pt'
    model_type = 'chemprop'
    save_path = data_path[:-4]+'_with_preds.csv'
    predict(data_path = data_path, model_path = model_path, model_type = model_type, save_path = save_path)
