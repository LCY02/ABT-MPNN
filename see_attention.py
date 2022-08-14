"""Visualize atomic attention and save similarity map."""

from typing import Union
import numpy as np
from tqdm import tqdm
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, update_prediction_args


def attention_visualization(args: PredictArgs):

    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])

    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    print('Loading data')
    full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[], ignore_columns=[],
                         skip_invalid_smiles=False, args=args, store_row=not args.drop_extra_columns)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i]
                                for i in sorted(full_to_valid_indices.keys())])

    print(f'Test size = {len(test_data):,}')

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Partial results for variance robust calculation.
    if args.ensemble_variance:
        all_preds = np.zeros(
            (len(test_data), num_tasks, len(args.checkpoint_paths)))

    print(
        f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        # Load model and scalers
        model = load_checkpoint(checkpoint_path, device=args.device)

    mpn = model.encoder

    for it, batch in enumerate(tqdm(test_data_loader, leave=False)):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch = batch.batch_graph()
        mol_adj_batch, mol_dist_batch, mol_clb_batch = batch.adj_features(
        ), batch.dist_features(), batch.clb_features()
        mpn.viz_attention(mol_batch, mol_adj_batch,
                          mol_dist_batch, mol_clb_batch, args.viz_dir)
    print('All done!')


if __name__ == "__main__":
    attention_visualization(args=PredictArgs().parse_args())
