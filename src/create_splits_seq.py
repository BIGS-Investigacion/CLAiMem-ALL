import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np
import json

#TODO: Configuration should be moved to a config file

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'task_3_tcga_breast_mollecular_subtyping', 'task_4_brca_breast_mollecular_subtyping'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the directory of the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--csv_path', type=str, default=None, 
                    help='manually specify the csv with the labels to use in classification (default: None)')
parser.add_argument('--label_dict', type=str, default=None, 
                    help='manually specify the labels associated with an index to be accessed (default: None)')
parser.add_argument('--patient_strat', action='store_true', default=False, help='forces isolated patients in the splits')

args = parser.parse_args()

json_acceptable_string = args.label_dict.replace("'", "\"")
labels = json.loads(json_acceptable_string)
if len(labels)==2:
    #args.n_classes=2
    #'data/dataset_csv/tumor_vs_normal_dummy_clean.csv'
    #{'normal_tissue':0, 'tumor_tissue':1}
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = labels,
                            patient_strat=args.patient_strat,
                            ignore=[],
                            balance_data=args.force_balance)

else: 
    #args.n_classes=3
    #'data/dataset_csv/tumor_subtyping_dummy_clean.csv'
    #{'subtype_1':0, 'subtype_2':1, 'subtype_3':2}
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = labels,
                            patient_strat= args.patient_strat,
                            patient_voting='maj',
                            ignore=[], 
                            balance_data=args.force_balance)

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.split_dir is None:
        split_dir = '.splits/'+ str(args.task)
    else:
        split_dir = args.split_dir
         
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        if len(label_fracs) > 1:
            split_dir = split_dir + '_{}'.format(int(lf * 100))
        if not os.path.isdir(split_dir):
            os.makedirs(split_dir, exist_ok=True)
            dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
            for i in range(args.k):
                dataset.set_splits()
                descriptor_df = dataset.test_split_gen(return_descriptor=True)
                splits = dataset.return_splits(from_id=True)
                save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
                save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
                descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



