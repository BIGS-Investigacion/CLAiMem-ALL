from __future__ import print_function
import argparse
import os
import sys
from pathlib import Path
import subprocess
import time
import shutil

# internal imports
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch

import pandas as pd
import numpy as np
import json


def create_aggregated_features(args, features_dir_original, output_base_dir, csv_file,
                               database_name, subtype_name, selection_method, top_k, n_splits):
    """
    Crea features agregadas usando atención

    Args:
        args: argumentos del parser
        features_dir_original: directorio con features originales
        output_base_dir: directorio base de salida
        csv_file: archivo CSV con labels
        database_name: nombre de la base de datos (cptac/tcga)
        subtype_name: nombre del subtipo (pam50/ihc/etc)
        selection_method: método de selección (self_attention, attention, norm, etc)
        top_k: número de features top a extraer
        n_splits: número de splits para dividir features

    Returns:
        output_dir: directorio donde se guardaron los features agregados
        csv_aggregated: path al CSV con labels de features agregados
    """
    output_dir = Path(output_base_dir) / database_name / subtype_name / "pt_files"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_pt_dir = Path(features_dir_original) / "pt_files"

    print(f"\n{'='*80}")
    print(f"STEP 1: PROCESSING FEATURES (ATTENTION-BASED AGGREGATION)")
    print(f"{'='*80}")
    print(f"Database:         {database_name}")
    print(f"Subtype:          {subtype_name}")
    print(f"Top-K:            {top_k}")
    print(f"Selection method: {selection_method}")
    print(f"N splits:         {n_splits}")
    print(f"Input dir:        {input_pt_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"{'='*80}\n")

    # Llamar al script de agregación
    cmd = [
        sys.executable,
        "src/claude/extract_top_attention_features.py",
        "--input_dir", str(input_pt_dir),
        "--output", str(output_dir),
        "--labels", csv_file,
        "--top_k", str(top_k),
        "--selection_method", selection_method,
        "--aggregation_method", "concat",
        "--n_splits", str(n_splits),
        "--save_metadata"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error processing features:")
        print(result.stderr)
        raise RuntimeError("Feature aggregation failed")

    print(result.stdout)

    # El CSV de salida se genera en output_dir/labels.csv
    csv_aggregated = output_dir / "labels.csv"

    if not csv_aggregated.exists():
        raise RuntimeError(f"Expected output CSV not found: {csv_aggregated}")

    return str(output_dir.parent), str(csv_aggregated)


def create_splits(args, csv_path, split_dir, label_dict_str, val_frac, test_frac, patient_strat):
    """
    Crea splits de train/val/test

    Args:
        args: argumentos del parser
        csv_path: path al CSV con labels
        split_dir: directorio donde guardar los splits
        label_dict_str: diccionario de labels como string
        val_frac: fracción de validación
        test_frac: fracción de test
        patient_strat: si usar estratificación por paciente
    """
    print(f"\n{'='*80}")
    print(f"CREATING SPLITS")
    print(f"{'='*80}")
    print(f"CSV path:     {csv_path}")
    print(f"Split dir:    {split_dir}")
    print(f"Val frac:     {val_frac}")
    print(f"Test frac:    {test_frac}")
    print(f"Patient strat: {patient_strat}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "src/create_splits_seq.py",
        "--seed", str(args.seed),
        "--k", str(args.k),
        "--test_frac", str(test_frac),
        "--val_frac", str(val_frac),
        "--split_dir", split_dir,
        "--label_frac", str(args.label_frac),
        "--csv_path", csv_path,
        "--label_dict", label_dict_str
    ]

    if patient_strat:
        cmd.append("--patient_strat")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error creating splits:")
        print(result.stderr)
        raise RuntimeError("Split creation failed")

    print(result.stdout)


def main(args, dataset_1: Generic_MIL_Dataset, dataset_2: Generic_MIL_Dataset):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []

    seed_torch(args.seed)
    train_dataset_1, val_dataset_1, _ = dataset_1.return_splits(from_id=False,
            csv_path='{}/splits_{}.csv'.format(args.split_dir_train, 0))

    datasets = (train_dataset_1, val_dataset_1, dataset_2)
    results, test_auc, val_auc, test_acc, val_acc  = train(datasets, 0, args)
    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)
    #write results to pkl
    filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(0))
    save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': 1, 'test_auc': all_test_auc,
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Generic training settings
    parser = argparse.ArgumentParser(description='Configurations for WSI Training with Attended Features')

    # Data settings
    parser.add_argument('--data_root_dir', type=str, default=None,
                        help='data directory')
    parser.add_argument('--embed_dim', type=int, default=1024)

    # Training settings
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--results_dir', default='./.results', help='results directory (default: ./.results)')

    # Split directories (will be auto-generated)
    parser.add_argument('--split_dir_train', type=str, default=None,
                        help='manually specify the set of splits to use in train (will be auto-generated if None)')
    parser.add_argument('--split_dir_test', type=str, default=None,
                        help='manually specify the set of splits to use in test (will be auto-generated if None)')

    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                        help='slide-level classification loss function (default: ce)')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb',
                        help='type of model (default: clam_sb, clam w/ single attention branch)')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping',
                                                      'task_3_tcga_breast_mollecular_subtyping',
                                                      'task_4_brca_breast_mollecular_subtyping'])

    # CSV paths
    parser.add_argument('--csv_path_train', type=str, default=None,
                        help='CSV with labels for training (original, before aggregation)')
    parser.add_argument('--csv_path_test', type=str, default=None,
                        help='CSV with labels for testing')
    parser.add_argument('--label_dict', type=str, default=None,
                        help='manually specify the labels associated with an index to be accessed (default: None)')

    # Data directories for features
    parser.add_argument('--data_root_dir_train_original', type=str, default=None,
                        help='data directory for the ORIGINAL training features (before aggregation)')
    parser.add_argument('--data_root_dir_test', type=str, default=None,
                        help='data directory for the test dataset (original features)')

    # CLAM settings
    parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                        help='disable instance-level clustering')
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                        help='instance-level clustering loss function (default: None)')
    parser.add_argument('--subtyping', action='store_true', default=False,
                        help='subtyping problem')
    parser.add_argument('--topo', action='store_true', default=False,
                     help='add topological diversity')
    parser.add_argument('--bag_weight', type=float, default=0.7,
                        help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')

    # NEW: Attention-based aggregation parameters
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top features to extract per WSI (default: 100)')
    parser.add_argument('--selection_method', type=str, default='self_attention',
                        choices=['attention', 'self_attention', 'norm', 'variance', 'random'],
                        help='Method for selecting top features (default: self_attention)')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='Number of files to generate per label in aggregation (default: 10)')

    # Database and subtype info (needed for creating aggregated features)
    parser.add_argument('--database_train', type=str, required=True,
                        choices=['cptac', 'tcga'],
                        help='Training database name')
    parser.add_argument('--database_test', type=str, required=True,
                        choices=['cptac', 'tcga'],
                        help='Test database name')
    parser.add_argument('--subtype', type=str, required=True,
                        help='Subtype name (pam50, ihc, ihc_simple, er, pr, erbb2)')
    parser.add_argument('--patient_strat', action='store_true', default=False,
                        help='Use patient stratification in splits')

    # Aggregated features output directory
    parser.add_argument('--aggregated_features_base_dir', type=str, default='data/aggregated_features',
                        help='Base directory for aggregated features output')

    args = parser.parse_args()

    json_acceptable_string = args.label_dict.replace("'", "\"")
    args.labels = json.loads(json_acceptable_string)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(args.seed)

    print(f"\n{'='*80}")
    print(f"HO_MAIN_ATTENDED - Training with Attention-Aggregated Features")
    print(f"{'='*80}")
    print(f"Train database: {args.database_train}")
    print(f"Test database:  {args.database_test}")
    print(f"Subtype:        {args.subtype}")
    print(f"Top-K:          {args.top_k}")
    print(f"Selection:      {args.selection_method}")
    print(f"N splits:       {args.n_splits}")
    print(f"{'='*80}\n")

    # STEP 1: Create aggregated features for TRAINING set
    timestamp_train = int(time.time())
    features_dir_train_agg, csv_train_agg = create_aggregated_features(
        args=args,
        features_dir_original=args.data_root_dir_train_original,
        output_base_dir=args.aggregated_features_base_dir,
        csv_file=args.csv_path_train,
        database_name=args.database_train,
        subtype_name=args.subtype,
        selection_method=args.selection_method,
        top_k=args.top_k,
        n_splits=args.n_splits
    )

    # STEP 2: Create splits for TRAINING (with aggregated features)
    if args.split_dir_train is None:
        args.split_dir_train = f".splits/{args.database_train}/ho-train-attended-{timestamp_train}"

    create_splits(
        args=args,
        csv_path=csv_train_agg,
        split_dir=args.split_dir_train,
        label_dict_str=args.label_dict,
        val_frac=0.15,  # Same as in bash script
        test_frac=0.0,
        patient_strat=args.patient_strat
    )

    # STEP 3: Create splits for TEST (with ORIGINAL features - no aggregation)
    timestamp_test = int(time.time())
    if args.split_dir_test is None:
        args.split_dir_test = f".splits/{args.database_test}/ho-test-attended-{timestamp_test}"

    create_splits(
        args=args,
        csv_path=args.csv_path_test,
        split_dir=args.split_dir_test,
        label_dict_str=args.label_dict,
        val_frac=0.0,  # No validation in test set
        test_frac=0.0,
        patient_strat=args.patient_strat
    )

    # Update results directory to include timestamps
    if args.results_dir == './.results':
        args.results_dir = (f".results/{args.database_train}/{args.subtype}/{args.model_type}/"
                          f"ho-attended-{'--patient_strat' if args.patient_strat else ''}-"
                          f"{args.database_train}-{timestamp_train}-{args.database_test}-{timestamp_test}/"
                          f"{args.exp_code}")

    settings = {'num_splits': args.k,
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs,
                'results_dir': args.results_dir,
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'model_type': args.model_type,
                'model_size': args.model_size,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                'labels':args.labels,
                'csv_path_train': csv_train_agg,  # Using aggregated CSV
                'csv_path_test': args.csv_path_test,
                'top_k': args.top_k,
                'selection_method': args.selection_method,
                'n_splits': args.n_splits
}

    if args.model_type in ['clam_sb', 'clam_mb']:
        settings.update({'bag_weight': args.bag_weight,
                            'inst_loss': args.inst_loss,
                            'B': args.B})

    print('\nLoad Dataset')

    if len(args.labels) >= 2:
        args.n_classes=len(args.labels)

        # TRAIN dataset: use AGGREGATED features
        dataset_1 = Generic_MIL_Dataset(csv_path = csv_train_agg,
                                data_dir= features_dir_train_agg,
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = args.labels,
                                patient_strat=False,
                                ignore=[])

        # TEST dataset: use ORIGINAL features (not aggregated)
        dataset_2 = Generic_MIL_Dataset(csv_path = args.csv_path_test,
                                data_dir= args.data_root_dir_test,
                                shuffle = False,
                                seed = args.seed,
                                print_info = True,
                                label_dict = args.labels,
                                patient_strat=False,
                                ignore=[])
    else:
        raise NotImplementedError

    if args.model_type in ['clam_sb', 'clam_mb']:
        assert args.subtyping

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    assert os.path.isdir(args.split_dir_train)
    assert os.path.isdir(args.split_dir_test)

    settings.update({'split_dir_train': args.split_dir_train})
    settings.update({'split_dir_test': args.split_dir_test})

    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))

    print(f"\n{'='*80}")
    print(f"STEP 3: TRAINING CLAM WITH AGGREGATED TRAIN FEATURES")
    print(f"{'='*80}")
    print(f"Train features:   {features_dir_train_agg} (AGGREGATED)")
    print(f"Test features:    {args.data_root_dir_test} (ORIGINAL)")
    print(f"Train CSV:        {csv_train_agg}")
    print(f"Test CSV:         {args.csv_path_test}")
    print(f"Results dir:      {args.results_dir}")
    print(f"Embed dim:        {args.embed_dim}")
    print(f"{'='*80}\n")

    results = main(args, dataset_1, dataset_2)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {args.results_dir}")
    print(f"{'='*80}")
    print("finished!")
    print("end script")
