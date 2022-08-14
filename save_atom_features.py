"""Computes and saves adjacency/distance/coulomb matrices for a dataset."""

import time
from tap import Tap

from chemprop.atomic_matrices import mol2matrix
from chemprop.features import save_features
from chemprop.data import get_smiles
from chemprop.utils import makedirs


class Args(Tap):
    data_path: str  # Path to data CSV
    # Name of the column containing SMILES strings. By default, uses the first column.
    smiles_column: str = None
    # Path to folder where features will be saved as a compressed numpy archive (.npz file )
    save_dir: str
    adjacency: bool = False  # Generate adjacency matrix
    coulomb: bool = False  # Generate coulomb matrix
    distance: bool = False  # Generate distance matrix


def generate_matrices(args: Args):
    """
    Computes and saves adjacency/distance/coulomb matrices for a dataset of molecules in a .npz file.

    :param args: Arguments.
    """
    # Create directory for save_dir
    makedirs(args.save_dir, isfile=True)

    smiles = get_smiles(path=args.data_path,
                        smiles_columns=args.smiles_column, flatten=True)
    print("Number of molecules: ", len(smiles))

    t = time.time()

    if args.coulomb or args.distance or args.adjacency:
        print("Generating atomic matrices")
        graph = mol2matrix(smiles, args)
        elapsed = time.time() - t

        if args.coulomb:
            clb = graph.get_coulomb()[1:]
            save_features(args.save_dir + "clb.npz", clb)
            print("Coulomb matrices saved.")

        if args.distance:
            dist = graph.get_distance()[1:]
            save_features(args.save_dir + "dist.npz", dist)
            print("Distance matrices saved.")

        if args.adjacency:
            adj = graph.get_adjacency()[1:]
            save_features(args.save_dir + "adj.npz", adj)
            print("Adjacency matrices saved.")

        print("Time:", elapsed)

    else:
        print("Please specify feature type from: (adjacency, distance, coulomb)")


if __name__ == '__main__':
    generate_matrices(Args().parse_args())
