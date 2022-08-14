from typing import List, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class MolMatrices:
    """
    A :class:`MolMatrices` represents the atomic matrices of a single molecule.

    A MolMatrices computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`f_adj`: The adjacency matrix of the molecule.
    * :code:`f_dist`: The distance matrix of the molecule.
    * :code:`f_clb`: The coulomb matrix of the molecule.
    """

    def __init__(self, mol: Union[str, Chem.Mol], args):

        self.smiles = mol
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        self.n_atoms = mol.GetNumAtoms()  # number of atoms

        # generate atomic matrices
        adj = np.zeros((self.n_atoms, self.n_atoms))  # Adjacency matrix
        clb = np.zeros((self.n_atoms, self.n_atoms))  # Coulomb matrix
        dis = np.zeros((self.n_atoms, self.n_atoms))  # Distance matrix

        try:
            AllChem.EmbedMolecule(mol)
            molh = Chem.AddHs(mol)
            self.conformer = molh.GetConformer(-1)

            for a1 in range(self.n_atoms):
                for a2 in range(self.n_atoms):
                    # Distance Matrix
                    if args.distance:
                        dis[a1, a2] = self.get_dist(self.conformer, a1, a2)

                    # Coulomb Matrix
                    if args.coulomb:
                        zi = mol.GetAtomWithIdx(a1).GetAtomicNum()
                        zj = mol.GetAtomWithIdx(a2).GetAtomicNum()
                        if a1 == a2:
                            clb[a1, a2] = 0.5 * zi ** 2.4
                        else:
                            conf_dist = self.get_dist(
                                self.conformer, a1, a2, clb=True)
                            if conf_dist == 0:
                                clb[a1, a2] == 0
                            else:
                                clb[a1, a2] = zi * zj / conf_dist

                     # Adjacency Matrix
                    if args.adjacency:
                        bond = mol.GetBondBetweenAtoms(a1, a2)
                        if bond is None:
                            adj[a1, a2] = 0
                        else:
                            adj[a1, a2] = 1

        except (ValueError, AttributeError, ZeroDivisionError):
            for a1 in range(self.n_atoms):
                for a2 in range(self.n_atoms):
                    # Distance Matrix
                    if args.distance:
                        dis[a1, a2] = 0

                    # Coulomb Matrix
                    if args.coulomb:
                        clb[a1, a2] = 0

                    # Adjacency Matrix
                    if args.adjacency:
                        bond = mol.GetBondBetweenAtoms(a1, a2)
                        if bond is None:
                            adj[a1, a2] = 0
                        else:
                            adj[a1, a2] = 1

        self.f_adj = adj
        self.f_dist = dis
        self.f_clb = clb

    def get_dist(self, mol, atomid1, atomid2, clb=False):
        position1 = np.array(mol.GetAtomPosition(atomid1))
        position2 = np.array(mol.GetAtomPosition(atomid2))
        if clb:
            distance = np.sqrt(np.sum((position1 - position2) ** 2))
        else:
            distance = np.linalg.norm(position1 - position2)
        return distance


class BatchMolMatrices:
    r"""
    :param mol_matrices: A list of :class:`MolMatrices`\ s from which to construct the :class:`BatchMolMatrices`.
    """

    def __init__(self, mol_matrices: List[MolMatrices]):

        self.smiles_batch = [mol_matrix.smiles for mol_matrix in mol_matrices]
        self.n_mols = len(self.smiles_batch)

        f_adj = [[]]
        f_dist = [[]]
        f_clb = [[]]

        for mol_matrix in mol_matrices:
            f_adj.append(mol_matrix.f_adj)
            f_dist.append(mol_matrix.f_dist)
            f_clb.append(mol_matrix.f_clb)

        self.f_adj = f_adj
        self.f_clb = f_clb
        self.f_dist = f_dist

    def get_adjacency(self):
        return self.f_adj

    def get_distance(self):
        return self.f_dist

    def get_coulomb(self):
        return self.f_clb


def mol2matrix(mols: Union[List[str], List[Chem.Mol]], args) -> BatchMolMatrices:

    return BatchMolMatrices([MolMatrices(mol, args) for mol in mols])
