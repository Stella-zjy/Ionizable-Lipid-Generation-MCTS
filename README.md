# Generative Model for Synthesizing Ionizable Lipids: A Monte Carlo Tree Search Approach

This project presents Monte Carlo Tree Search (MCTS)-based generative models designed to create novel ionizable lipids with synthetic feasibility and desirable biomedical properties. This research is based on my Master’s thesis: Generative Models for Synthesizable Lipids, supervised by Professor José Miguel Hernández-Lobato at the University of Cambridge.

This project has been accepted at the NeurIPS 2024 AI for New Drug Modalities Workshop.

## Credits

This project builds upon previous works in the field of generative models and cheminformatics. I would like to acknowledge the following foundational research and tools that inspired and supported this project:

- **MCTS for Molecular Design**: Our project builds upon the [SyntheMol](https://github.com/swansonk14/SyntheMol) project, which also serves as our baseline model.
- **pKa Prediction**: We make use of the [MolGpKa](https://github.com/Xundrug/MolGpKa) module for pKa prediction, which serves as an important component in our ionizability predictor.

## Getting Started
### Prerequisites

Ensure the following dependencies are installed:

```bash
conda create --name mcts-lipid python=3.10
conda activate mcts-lipid
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install rdkit
pip install py3Dmol
pip install numpy pandas scikit-learn
pip install matplotlib
pip install scipy tqdm
pip install chemfunc==1.0.3
pip install chemprop==1.6.1
pip install typed-argument-parser==1.8.0
```

For running the MolGpKa module on a GPU (with CUDA 11.8), use the following:

```bash
conda create --name molgpka-gpu python=3.10
conda activate molgpka-gpu
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install rdkit
pip install py3Dmol
pip install numpy pandas scikit-learn matplotlib
```

### Instructions

#### Building Block Dataset

Building block dataset creation is a collaborative effort. The source code for this part can be viewed [here](https://github.com/yuxuanou623/Lipid_reaction_dataset_creation).

#### Lipid Classifier

We use a Chemprop-based model for lipid classification, utilizing the same training script from SyntheMol. To train the lipid classifier, run the following command:

```bash
python scripts/models/train.py \
    --data_path data/chemprop_training_data.csv \
    --save_dir data/Models/lipid_classifier_chemprop_1_epochs \
    --dataset_type classification \
    --model_type chemprop \
    --property_column target \
    --num_models 1 \
    --epochs 1
```

#### Ionizability Predictor

For pKa prediction, we utilize the `MolGpKa` module. The ionizability filtering criteria are implemented in the `MolGpKa/src/charge_ph.py` file. Specifically, the `ionizability_classifier_single_molecule()` function is used to evaluate generated molecules.

#### Reaction Prediction

We employ a template-based reaction prediction model, using the same reactions defined in SyntheMol. Reaction templates are located in the `synthemol/reactions/real.py` file. You can easily substitute these with custom reaction templates if needed.

#### Baseline (SyntheMol)

The SyntheMol approach for lipid generation can be experimented by running the following script:
```bash
python synthemol/generate/generate.py
```
This script is adapted from the original source code provided by SyntheMol.

#### Policy Network Guided MCTS

The scripts for policy network-guided MCTS are located in the `synthemol/guided_mcts` directory. To experiment with the guided MCTS approach, run the following command:
```bash
python synthemol/guided_mcts/run.py
```

