from __future__ import print_function
import argparse
import os

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
    #datasets = (dataset_1, dataset_1, dataset_2)
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
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_root_dir', type=str, default=None, 
                        help='data directory')
    parser.add_argument('--embed_dim', type=int, default=1024)
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
    parser.add_argument('--split_dir_train', type=str, default=None, 
                        help='manually specify the set of splits to use in train, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--split_dir_test', type=str, default=None, 
                        help='manually specify the set of splits to use in train, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                        help='slide-level classification loss function (default: ce)')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'abmil', 'rrt', 'transmil', 'wikg'], default='clam_sb',
                        help='type of model (default: clam_sb, clam w/ single attention branch)')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'task_3_tcga_breast_mollecular_subtyping', 'task_4_brca_breast_mollecular_subtyping'])
    parser.add_argument('--csv_path_train', type=str, default=None, 
                        help='manually specify the csv with the labels to use in classification training (default: None)')
    parser.add_argument('--csv_path_test', type=str, default=None, 
                        help='manually specify the csv with the labels to use in classification test (default: None)')
    parser.add_argument('--label_dict', type=str, default=None, 
                        help='manually specify the labels associated with an index to be accessed (default: None)')
    parser.add_argument('--data_root_dir_train', type=str, default=None, 
                        help='data directory for the training dataset')
    parser.add_argument('--data_root_dir_test', type=str, default=None, 
                        help='data directory for the test dataset')
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
    parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
    ### MIL techniques options (applies to abmil, rrt, transmil, wikg)
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension for MIL techniques (default: 512)')
    parser.add_argument('--abmil_is_norm', action='store_true', default=True, help='abmil: normalize attention weights (default: True)')


    args = parser.parse_args()

    json_acceptable_string = args.label_dict.replace("'", "\"")
    args.labels = json.loads(json_acceptable_string)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch(args.seed)

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
                'csv_path_train': args.csv_path_train,
                'csv_path_test': args.csv_path_test
}

    if args.model_type in ['clam_sb', 'clam_mb']:
        settings.update({'bag_weight': args.bag_weight,
                            'inst_loss': args.inst_loss,
                            'B': args.B})



    print('\nLoad Dataset')

    #TODO: To be fixed. Config file should be used to execute tasks



    if len(args.labels) >= 2:
        args.n_classes=len(args.labels)
        dataset_1 = Generic_MIL_Dataset(csv_path = args.csv_path_train,
                                data_dir= args.data_root_dir_train,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = args.labels,
                                patient_strat=False,
                                ignore=[])        
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

    #args.results_dir = os.path.join(args.results_dir, str(datetime.timestamp(datetime.now())) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.split_dir_train is None:
        args.split_dir_train = os.path.join('.splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir_train = args.split_dir_train

    if args.split_dir_test is None:
        args.split_dir_test = os.path.join('.splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir_test = args.split_dir_test


    assert os.path.isdir(args.split_dir_train)
    assert os.path.isdir(args.split_dir_train)

    settings.update({'split_dir_train': args.split_dir_train})
    settings.update({'split_dir_test': args.split_dir_test})


    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))     
    results = main(args, dataset_1, dataset_2)
    print("finished!")
    print("end script")


