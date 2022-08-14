import os

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt

from .nn_utils import index_select_ND


def visualize_atom_attention(viz_dir: str,
                             smiles: str,
                             num_atoms: int,
                             attention_weights: torch.FloatTensor):
    """
    Saves figures of attention maps between atoms. Note: works on a single molecule, not in batch
    :param viz_dir: Directory in which to save attention map figures.
    :param smiles: Smiles string for molecule.
    :param num_atoms: The number of atoms in this molecule.
    :param attention_weights: A num_atoms x num_atoms PyTorch FloatTensor containing attention weights.
    """
    if type(smiles) == str:
        mol_name = smiles
        print('Saving {0} ({1} atoms)'.format(smiles, num_atoms))
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
        mol_name = Chem.MolToSmiles(mol)
        print('Saving Similarity map of molecule: {0} ({1} atoms)'.format(
            mol_name, num_atoms))

    smiles_viz_dir = viz_dir
    os.makedirs(smiles_viz_dir, exist_ok=True)
    atomSum_weights = np.zeros(num_atoms)

    # attention_weights = (attention_weights[4] + attention_weights[5]) / 2

    for a in range(num_atoms):
        a_weights = attention_weights[a].cpu().data.numpy()
        atomSum_weights += a_weights

    Amean_weight = atomSum_weights / num_atoms

    nanMean = np.nanmean(Amean_weight)

    save_path = os.path.join(
        smiles_viz_dir, f'{mol_name.replace("/", "")}.png')

    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, Amean_weight-nanMean,
                                                     alpha=0.3,
                                                     size=(300, 300))

    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


# Todo
def visualize_bond_attention(viz_dir: str,
                             mol_graph: None,
                             depth: int,
                             attention_weights: torch.FloatTensor):
    """
    Saves figures of attention maps between bonds.
    :param viz_dir: Directory in which to save attention map figures.
    :param mol_graph: BatchMolGraph containing a batch of molecular graphs.
    :param attention_weights: A num_bonds x num_bonds PyTorch FloatTensor containing attention weights.
    :param depth: The current depth (i.e. message passing step).
    """
    for i in range(mol_graph.n_mols):
        smiles = mol_graph.smiles_batch[i]

        if type(smiles) == str:
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = smiles

        smiles_viz_dir = os.path.join(viz_dir, 'Bond')
        os.makedirs(smiles_viz_dir, exist_ok=True)

        a_start, a_size = mol_graph.a_scope[i]
        b_start, b_size = mol_graph.b_scope[i]
        atomSum_weights = np.zeros(a_size)

        for b in range(b_start, b_start + b_size):

            a1 = mol_graph.b2a[b].item() - a_start
            a2 = mol_graph.b2a[mol_graph.b2revb[b]].item() - a_start

            b_weights = attention_weights[i][b].cpu().data  # 1 x num_bonds

            a2b = mol_graph.a2b[a_start: a_start + a_size].cpu().data

            a_weights = index_select_ND(b_weights, a2b[i])
            a_weights = a_weights.sum(dim=1)
            a_weights = a_weights.cpu().data.numpy()
            atomSum_weights += a_weights
        Amean_weight = atomSum_weights / a_size
        nanMean = np.nanmean(Amean_weight)

        save_path = os.path.join(
            smiles_viz_dir, f'{Chem.MolToSmiles(mol)}.png')

        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, Amean_weight-nanMean,
                                                         alpha=0.3,
                                                         size=(300, 300))
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
