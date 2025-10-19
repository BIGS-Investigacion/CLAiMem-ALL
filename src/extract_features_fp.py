import time
import os
import argparse
from dotenv import load_dotenv
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm

import numpy as np
import torchstain

from bigs_auxiliar.normalizers import MacenkoNormalizer
from models.virtual_stainer import BCIEvaluatorBasicExt
from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Virtual_Whole_Slide_Bag_FP, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

def auxiliar(dataset_, batch_size_, output_path_, model_, loader_kwargs, time_start):
	features = []
	loader = DataLoader(dataset=dataset_, batch_size=batch_size_, **loader_kwargs)
	output_file_path = compute_w_loader(output_path_, loader = loader, model = model_, verbose = 1)

	time_elapsed = time.time() - time_start
	print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

	with h5py.File(output_file_path, "r") as file:
		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
	return features


# ============================================================================
# ARGUMENTOS
# ============================================================================

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 
																				 'uni_v1', 'uni_v2', 
																				 'conch_v1', 'ctranspath', 
																				 'retccl', 'provgigapath', 
																				 'phikon', 'musk', 
																				 'virchow', 'hoptimus0', 
																				 'hibou_l', 'hibou_b'])	
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--virtual', default=False, action='store_true')

# Argumentos para Macenko normalization
parser.add_argument('--use_macenko', default=False, action='store_true',
					help='Apply Macenko stain normalization during feature extraction')
parser.add_argument('--reference_image', type=str, default=None,
					help='Path to Macenko reference image (.npy). Required if --use_macenko is set')

args = parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
	
	# ========================================================================
	# Inicializar Macenko normalizer si es necesario
	# ========================================================================
	normalizer = None
	if args.use_macenko:
		print("\n" + "="*70)
		print("MACENKO STAIN NORMALIZATION ENABLED")
		print("="*70)
		
		if args.reference_image is None:
			raise ValueError(
				"❌ --reference_image is required when using --use_macenko\n"
				"   Please provide the path to the Macenko reference .npy file"
			)
		
		if not os.path.exists(args.reference_image):
			raise ValueError(f"❌ Reference image not found: {args.reference_image}")
		
		print(f"\n📸 Loading Macenko reference from: {args.reference_image}")
		ref_image = np.load(args.reference_image)
		print(f"   Reference shape: {ref_image.shape}")
		
		# Inicializar con verbose=False para evitar demasiados prints
		normalizer = MacenkoNormalizer()
		normalizer.fit(ref_image)
		print(f"✓ Macenko normalizer fitted and ready")
		print(f"   Min tissue percentage: 5% (patches with less tissue will be skipped)")
		print("="*70 + "\n")
	
	# ========================================================================
	# Inicialización del dataset
	# ========================================================================
	
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == "cuda" else {}
	
	if args.virtual:
		# configurations of experiment
		load_dotenv()
		# settings
		apply_tta=os.getenv('BCI_STAINER_APPLY_TTA')
		model_name=os.getenv('BCI_STAINER_MODEL')
		config_file=os.getenv('BCI_STAINER_CONFIG') 
		exp_root=os.getenv('BCI_STAINER_MODELS_DIR')   
		
		# loads configs
		configs = OmegaConf.load(config_file)
		model_path = os.path.join(exp_root, configs.exp, f'{model_name}.pth')

		evaluator = BCIEvaluatorBasicExt(configs, model_path, apply_tta)

	# ========================================================================
	# Procesamiento de cada slide
	# ========================================================================
	
	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		try:
			output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
			time_start = time.time()
			wsi = openslide.open_slide(slide_file_path)
			
			# CRÍTICO: Pasar normalizer al dataset
			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
										wsi=wsi, 
										img_transforms=img_transforms,
										normalizer=normalizer)  # ← Pasar normalizer

			features = auxiliar(dataset, args.batch_size, output_path, model, loader_kwargs, time_start)
			
			if args.virtual:
				dataset = Virtual_Whole_Slide_Bag_FP(file_path=h5_file_path, 
										wsi=wsi,
										virtualizer=evaluator, 
										img_transforms=img_transforms,
										normalizer=normalizer)  # ← Pasar normalizer también aquí
				virtual_features = auxiliar(dataset, args.batch_size, output_path, model, loader_kwargs, time_start)
				features = np.concatenate((features, virtual_features), axis=1)  
			
			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)
			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
		except Exception as e:
			print(e)
