from typing import List, Union
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .attention import MultiBondAttention, MultiBondFastAttention, MultiAtomAttention, SublayerConnection
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function
from chemprop.attention_visualization import visualize_atom_attention


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        # Centers messages on atoms instead of on bonds. (bool = False)
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth  # Number of message passing steps.
        self.dropout = args.dropout

        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        # Literal['mean', 'sum', 'norm'] = 'mean'
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm  # if aggregation == 'mean'

        self.bond_attention = args.bond_attention
        self.bond_fast_attention = args.bond_fast_attention
        self.atom_attention = args.atom_attention
        self.distance = args.distance
        self.adjacency = args.adjacency
        self.coulomb = args.coulomb

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        # input_dim = bond_fdim = 147
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size,
                             self.hidden_size)  # contains bias

        if self.bond_attention:
            self.bond_attention_block = MultiBondAttention(args)
            self.bond_residual = SublayerConnection(dropout=self.dropout)

        if self.bond_fast_attention:
            self.bond_attention_block = MultiBondFastAttention(args)
            self.bond_residual = SublayerConnection(dropout=self.dropout)

        if self.atom_attention:
            self.atom_attention_block = MultiAtomAttention(args)
            self.atom_residual = SublayerConnection(dropout=self.dropout)

    def forward(self,
                mol_graph: BatchMolGraph,
                mol_adj_batch: List[np.ndarray] = None,
                mol_dist_batch: List[np.ndarray] = None,
                mol_clb_batch: List[np.ndarray] = None,
                viz_dir: str = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param mol_adj_batch: A list of numpy arrays containing additional adjacency matrices
        :param mol_dist_batch: A list of numpy arrays containing additional distance matrices
        :param mol_clb_batch: A list of numpy arrays containing additional coulomb matrices
        :param viz_dir: Path to CSV file where similarity maps will be saved
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

        # batch_n_atoms x n_atoms
        f_adj = mol_adj_batch
        f_dist = mol_dist_batch
        f_clb = mol_clb_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(
            atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(
            self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)
            # input: batch_num_bonds x hidden_size

        # message: batch_num_bonds x hidden_size
        message = self.act_func(input)

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                # num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2a)
                # num_atoms x max_num_bonds x bond_fdim
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                # num_atoms x max_num_bonds x hidden + bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                # num_atoms x hidden + bond_fdim
                message = nei_message.sum(dim=1)

            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                # batch_num_atoms x max_num_bonds x hidden
                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(
                    dim=1)  # batch_num_atoms x hidden
                rev_message = message[b2revb]  # batch_num_bonds x hidden
                # batch_num_bonds x hidden
                message = a_message[b2a] - rev_message
                if self.bond_attention or self.bond_fast_attention:
                    att_message = self.bond_attention_block(message, b_scope)
                    # message: batch_num_bonds x hidden_size
                    message = self.bond_residual(message, att_message)

            message = self.W_h(message)
            # batch_num_bonds x hidden_size
            message = self.act_func(input + message)
            message = self.dropout_layer(message)  # batch_num_bonds x hidden

        # atom level
        a2x = a2a if self.atom_messages else a2b

        # batch_num_atoms x max_num_bonds x hidden
        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)  # batch_num_atoms x hidden

        # batch_num_atoms x (atom_fdim + hidden)
        a_input = torch.cat([f_atoms, a_message], dim=1)

        # batch_num_atoms x hidden
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(
            atom_hiddens)  # batch_num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):  # a_scope: (1, num_atoms)
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                # num_atoms x hidden + bond_fdim
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)

                if self.atom_attention:
                    mol_vec, att_a_w = self.atom_attention_block(
                        cur_hiddens, i, f_adj, f_dist, f_clb, viz_dir)
                    mol_vec = self.atom_residual(cur_hiddens, mol_vec)

                    if viz_dir:
                        visualize_atom_attention(
                            viz_dir, mol_graph.smiles_batch[i], a_size, att_a_w)

                else:
                    mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size  # (1, hidden_size)
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(
            overwrite_default_atom=args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList(
                [MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])  # args.number_of_molecules = 1

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                mol_adj_batch: List[np.ndarray] = None,
                mol_dist_batch: List[np.ndarray] = None,
                mol_clb_batch: List[np.ndarray] = None,
                ) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list is of length :code:`number_of_molecules` (number of molecules per datapoint),
                      the inner list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch).
        :param features_batch: A list of numpy arrays containing additional features.
        :param mol_adj_batch: A list of numpy arrays containing additional adjacency matrices
        :param mol_dist_batch: A list of numpy arrays containing additional distance matrices
        :param mol_clb_batch: A list of numpy arrays containing additional coulomb matrices
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

        if type(batch[0]) != BatchMolGraph:
            print('type(batch[0]) != BatchMolGraph')
            batch = [mol2graph(b, mol_adj_batch, mol_dist_batch,
                               mol_clb_batch) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(
                np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        encodings = [enc(ba, mol_adj_batch, mol_dist_batch, mol_clb_batch)
                     for enc, ba in zip(self.encoder, batch)]
        output = reduce(lambda x, y: torch.cat(
            (x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output

    def viz_attention(self,
                      batch: Union[List[str], BatchMolGraph],
                      mol_adj_batch: List[np.ndarray] = None,
                      mol_dist_batch: List[np.ndarray] = None,
                      mol_clb_batch: List[np.ndarray] = None,
                      viz_dir: str = None):

        encodings = [enc(ba, mol_adj_batch, mol_dist_batch, mol_clb_batch, viz_dir=viz_dir)
                     for enc, ba in zip(self.encoder, batch)]
