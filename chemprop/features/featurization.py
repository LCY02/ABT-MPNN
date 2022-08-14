from typing import List, Tuple, Union
from itertools import zip_longest
from rdkit import Chem
import torch
import numpy as np

# Atom feature sizes (atom_fdim: 127)
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    # type of atom (ex. C,N,O), by atomic number, size = 100
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    # number of bonds the atom is involved in, size = 6
    'degree': [0, 1, 2, 3, 4, 5],
    # integer electronic charge assigned to atom, size = 5
    'formal_charge': [-1, -2, 1, 2, 0],
    # chirality: unspecified, tetrahedral CW/CCW, or other, size = 4
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],  # number of bonded hydrogen atoms, size = 5
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(
    range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# FDIM: feature dimensionality
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
EXTRA_ATOM_FDIM = 0
BOND_FDIM = 14
EXTRA_BOND_FDIM = 0


def get_atom_fdim(overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the atom feature vector.
    """
    return (not overwrite_default_atom) * ATOM_FDIM + EXTRA_ATOM_FDIM


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    global EXTRA_ATOM_FDIM
    EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the bond feature vector.
    """

    return (not overwrite_default_bond) * BOND_FDIM + EXTRA_BOND_FDIM + \
           (not atom_messages) * \
        get_atom_fdim(overwrite_default_atom=overwrite_default_atom)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    global EXTRA_BOND_FDIM
    EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)  # create the list with zeros
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    # feature vector for each atom
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
        onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
        onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
        onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
        onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
        onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
        [1 if atom.GetIsAromatic() else 0] + \
        [atom.GetMass() * 0.01]  # scaled to about the same range as other features

    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


class MolGraph:
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`f_adj`: The adjacency matrix of the molecule.
    * :code:`f_dist`: The distance matrix of the molecule.
    * :code:`f_clb`: The coulomb matrix of the molecule.
    """

    def __init__(self, mol: Union[str, Chem.Mol],
                 adj: np.ndarray = None,
                 dist: np.ndarray = None,
                 clb: np.ndarray = None):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param f_adj: The adjacency matrix of the molecule.
        :param f_dist: The distance matrix of the molecule.
        :param f_clb: The coulomb matrix of the molecule.
        """

        self.smiles = mol
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        # mapping from bond index to concat(in_atom, bond) features
        self.f_bonds = []
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        # Get atom features
        # atom feature (size = 133)
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
        self.f_adj = adj
        self.f_dist = dist
        self.f_clb = clb

        self.n_atoms = len(self.f_atoms)  # Initialize number of atoms

        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        # iterate throgh all the bonds (based on every two adjacent atoms)
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                # if there's no bond between them, then continue
                if bond is None:
                    continue

                # bond feature (the bond between a1 and a2)
                f_bond = bond_features(bond)

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                # a2 atom with adjacent bond (size = 147)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                # b1 = a1 --> a2   the atom (a2) has the incoming bond (b1)
                self.a2b[a2].append(b1)
                # the bond (b1) is originated from atom (a1)
                self.b2a.append(a1)
                # b2 = a2 --> a1   the atom (a1) has the incoming bond (b2)
                self.a2b[a1].append(b2)
                # the bond (b2) is originated from atom (a2)
                self.b2a.append(a2)
                self.b2revb.append(b2)  # first append b2
                self.b2revb.append(b1)  # then append b1
                self.n_bonds += 2


class BatchMolGraph:

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim()

        # Start n_atoms and n_bonds at 1 b/c zero padding
        # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_atoms = 1
        # number of bonds (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1
        # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.a_scope = []
        # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.b_scope = []

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        f_adj = [[]]
        f_dist = [[]]
        f_clb = [[]]
        a2b = [[]]  # mapping from atom index to incoming bond indices
        # mapping from bond index to the index of the atom the bond is coming from
        b2a = [0]
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:  # for each molecule graph
            f_atoms.extend(mol_graph.f_atoms)  # n_atoms * 133
            f_bonds.extend(mol_graph.f_bonds)  # n_bonds * 147
            f_adj.append(mol_graph.f_adj)  # n_atoms * n_atoms
            f_dist.append(mol_graph.f_dist)
            f_clb.append(mol_graph.f_clb)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])
                # f_adj.append(mol_graph.f_adj) # n_atoms * n_atoms

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.f_adj = f_adj  # batch_n_atoms * n_atoms
        self.f_clb = f_clb
        self.f_dist = f_dist
        self.a2b = torch.LongTensor(
            [a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        if atom_messages:
            f_bonds = self.f_bonds[:, -
                                   get_bond_fdim(atom_messages=atom_messages):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    # try to avoid computing b2b b/c O(n_atoms^3)
    def get_b2b(self) -> torch.LongTensor:
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(
                1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    # only needed if using atom messages
    def get_a2a(self) -> torch.LongTensor:
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds
        return self.a2a

    def get_adjacency(self):
        return self.f_adj

    def get_distance(self):
        return self.f_dist

    def get_coulomb(self):
        return self.f_clb


def mol2graph(mols: Union[List[str], List[Chem.Mol]],
              mol_adj_batch: List[np.array] = None,
              mol_dist_batch: List[np.array] = None,
              mol_clb_batch: List[np.array] = None,
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param mol_adj_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param mol_dist_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param mol_dist_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol, adj, dist, clb)
                          for mol, adj, dist, clb
                          in zip_longest(mols, mol_adj_batch, mol_dist_batch, mol_clb_batch)])
