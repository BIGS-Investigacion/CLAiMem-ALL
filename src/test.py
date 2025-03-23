import torch

from dataset_modules.dataset_generic import Generic_MIL_Dataset, Generic_Split
import numpy as np

#model_path = '/media/jorge/investigacion/software/CLAiMemAll/.results/cptac/er/10-cv_1741148700/clam_sb/uni/s_4_checkpoint.pt'
'''tecnica='phikon'
data = '/media/jorge/SP_PHD_U3/perfil_molecular/features/cptac/features_'+tecnica+'/pt_files/01BR001-4ffefc66-d0ba-4a36-b4fa-35bd91.pt'
try:
    model = torch.load(data)
    model = model.view(model.size(0), -1)
    print(model.size())
except RuntimeError as e:
    print(f"Failed to load model: {e}")'''


'''dataset = Generic_MIL_Dataset(csv_path = 'data/dataset_csv/cptac-subtype_pam50.csv','
'
                            data_dir= '/media/jorge/SP_PHD_U3/perfil_molecular/features/cptac/features_cnn',
                            shuffle = False, 
                            seed = 42, 
                            print_info = True,
                            label_dict = {'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4},
                            patient_strat=False,
                            ignore=[])
dataset.train_ids = [i for i in range(387)]
dataset.val_ids = []
dataset.test_ids = []

train_dataset, _, _ = dataset.return_splits()

# Example list of numpy arrays
arrays = train_dataset.slide_cls_ids
print(arrays)'''