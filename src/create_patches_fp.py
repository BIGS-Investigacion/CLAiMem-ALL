# internal imports
from bigs_auxiliar.normalizers import MacenkoNormalizer
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torchstain
import cv2
from PIL import Image
import shutil
import h5py




# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
	"""
	Patching sin pasar normalizer (ya está en WSI_object.normalizer)
	"""
	### Start Patch Timer
	start_time = time.time()

	# Patch (normalizer ya está en WSI_object)
	file_path = WSI_object.process_contours(**kwargs)

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


# ============================================================================
# FUNCIÓN PRINCIPAL seg_and_patch
# ============================================================================

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None,
				  normalizer=None):
	
	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Initialize WSI
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)
		
		# CRÍTICO: Asignar normalizer como ATRIBUTO del objeto
		if normalizer is not None:
			WSI_object.normalizer = normalizer

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}

			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir})
			# NO pasar normalizer (ya está en WSI_object.normalizer)
			file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	if total>0:	
		seg_times /= total
		patch_times /= total
		stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times


# ============================================================================
# ARGUMENTOS
# ============================================================================

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

# Argumentos para stain normalization
parser.add_argument('--use_macenko', default=False, action='store_true',
					help='Apply Macenko stain normalization (saves to separate directory with _macenko suffix)')
parser.add_argument('--reference_image', type=str, default=None,
					help='Path to reference image (.npy) for stain normalization. If not provided, uses slide with most tissue as reference')
parser.add_argument('--original_masks_dir', type=str, default=None,
					help='Path to original masks directory (for loading segmentations when using Macenko). If not provided, will look in save_dir without _macenko suffix')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
	args = parser.parse_args()

	# ========================================================================
	# CRÍTICO: Si se usa Macenko, añadir sufijo a los directorios de salida
	# ========================================================================
	if args.use_macenko:
		# Añadir sufijo '_macenko' al directorio de salida
		base_save_dir = args.save_dir
		if not base_save_dir.endswith('_macenko'):
			args.save_dir = base_save_dir.rstrip('/') + '_macenko'
			print("\n" + "="*70)
			print("⚠   MACENKO MODE: Output will be saved to a separate directory")
			print(f"   Original: {base_save_dir}")
			print(f"   Macenko:  {args.save_dir}")
			print("="*70 + "\n")

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)
	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)
	
	# ========================================================================
	# Si usa Macenko, copiar máscaras del directorio original
	# ========================================================================
	if args.use_macenko:
		# Determinar directorio de máscaras originales
		if args.original_masks_dir is not None:
			original_mask_dir = args.original_masks_dir
		else:
			original_save_dir = args.save_dir.replace('_macenko', '')
			original_mask_dir = os.path.join(original_save_dir, 'masks')
		
		if os.path.exists(original_mask_dir) and os.path.isdir(original_mask_dir):
			print(f"\n📋 Copying segmentation masks from {original_mask_dir} to {mask_save_dir}...")
			mask_files = [f for f in os.listdir(original_mask_dir) if f.endswith('.pkl')]
			
			if len(mask_files) > 0:
				for mask_file in mask_files:
					src = os.path.join(original_mask_dir, mask_file)
					dst = os.path.join(mask_save_dir, mask_file)
					if not os.path.exists(dst):
						shutil.copy2(src, dst)
				print(f"✓ Copied {len(mask_files)} segmentation mask(s)")
			else:
				print(f"⚠  Warning: No .pkl mask files found in {original_mask_dir}")
		else:
			print(f"⚠  Warning: Original mask directory not found: {original_mask_dir}")
			print("   Segmentation will be performed from scratch.")

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('config/presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	# ========================================================================
	# Inicializar Macenko normalizer
	# ========================================================================
	
	normalizer = None
	if args.use_macenko:
		print("\n" + "="*70)
		print("MACENKO STAIN NORMALIZATION ENABLED")
		print("="*70)
		
		normalizer = MacenkoNormalizer()
		
		# Si no hay referencia, crear una con la WSI que tenga más tejido
		if args.reference_image is None:
			print("No reference image provided. Finding slide with most tissue content...")
			
			# MEJOR MÉTODO: Usar archivos .h5 que contienen coordenadas EXACTAS de patches
			original_save_dir = args.save_dir.replace('_macenko', '')
			original_patch_dir = os.path.join(original_save_dir, 'patches')
			
			# Obtener lista de slides
			all_slides = sorted([s for s in os.listdir(args.source) 
								if os.path.isfile(os.path.join(args.source, s))])
			
			if len(all_slides) == 0:
				raise ValueError("No slides found in source directory")
			
			slides_to_check = all_slides[:min(10, len(all_slides))]
			
			best_slide = None
			max_tissue_patches = 0
			best_patch_coords = None
			
			if os.path.exists(original_patch_dir):
				print(f"✓ Found patches directory: {original_patch_dir}")
				print(f"Analyzing {len(slides_to_check)} slides to find best reference...")
				
				for slide_name in tqdm(slides_to_check, desc="Finding best reference"):
					slide_id = os.path.splitext(slide_name)[0]
					h5_file = os.path.join(original_patch_dir, slide_id + '.h5')
					
					print(f"\n  {slide_name}:")
					
					if not os.path.exists(h5_file):
						print(f"    ⚠  No .h5 file found, skipping...")
						continue
					
					try:
						# Leer archivo .h5 para obtener coordenadas de patches
						with h5py.File(h5_file, 'r') as f:
							coords = f['coords'][:]
							num_patches = len(coords)
							
							print(f"    ✓ Found {num_patches} patches")
							
							if num_patches > max_tissue_patches:
								max_tissue_patches = num_patches
								best_slide = slide_name
								# Seleccionar un patch del medio (más probable que tenga tejido representativo)
								middle_idx = num_patches // 2
								best_patch_coords = coords[middle_idx]
								print(f"    ✓ New best! Patch at coords: {best_patch_coords}")
					
					except Exception as e:
						print(f"    ✗ Error reading .h5: {e}")
						continue
				
				if best_slide is None or best_patch_coords is None:
					raise ValueError(
						f"❌ No valid .h5 files found in {original_patch_dir}!\n"
						"Please run patching first WITHOUT --use_macenko:\n"
						f"  python create_patches_fp.py --source {args.source} --save_dir {original_save_dir} --seg --patch --stitch\n"
						"Then re-run with --use_macenko"
					)
				
				print(f"\n✓ Selected {best_slide} as reference ({max_tissue_patches} patches)")
				print(f"  Using patch at coordinates: {best_patch_coords}")
				
				# Cargar WSI correspondiente
				slide_path = os.path.join(args.source, best_slide)
				wsi_obj = WholeSlideImage(slide_path)
				wsi = wsi_obj.getOpenSlide()
				
				print(f"\n📸 Extracting reference patch from WSI...")
				
				# Extraer el patch usando las coordenadas exactas del .h5
				x, y = best_patch_coords
				patch_size = args.patch_size
				level = args.patch_level
				
				# Leer región
				ref_region = wsi.read_region((int(x), int(y)), level, (patch_size, patch_size))
				ref_region_rgb = np.array(ref_region.convert('RGB'))
				
				print(f"  ✓ Extracted {patch_size}x{patch_size} patch at level {level}")
				print(f"    Coordinates: ({x}, {y})")
				
				# VERIFICAR que la región tiene contenido
				gray_check = cv2.cvtColor(ref_region_rgb, cv2.COLOR_RGB2GRAY)
				non_background = np.sum((gray_check > 10) & (gray_check < 245))
				total_pixels = patch_size * patch_size
				tissue_percentage = non_background / total_pixels
				
				print(f"  Region statistics:")
				print(f"    Tissue pixels: {non_background:,} / {total_pixels:,} ({tissue_percentage:.1%})")
				print(f"    Mean intensity: {gray_check.mean():.1f}")
				print(f"    Intensity range: [{gray_check.min()}, {gray_check.max()}]")
				
				if tissue_percentage < 0.3:
					print(f"\n⚠  Warning: Selected patch has low tissue content ({tissue_percentage:.1%})")
					print(f"  You may want to manually specify --reference_image")
				
			else:
				raise ValueError(
					f"❌ Patches directory not found: {original_patch_dir}\n"
					"Please run patching first WITHOUT --use_macenko:\n"
					f"  python create_patches_fp.py --source {args.source} --save_dir {original_save_dir} --seg --patch --stitch\n"
					"Then re-run with --use_macenko"
				)
			
			# Fit normalizer
			normalizer.fit(ref_region_rgb)
			
			# Guardar referencia para reutilizar
			ref_save_path = os.path.join(args.save_dir, 'macenko_reference.npy')
			np.save(ref_save_path, ref_region_rgb)
			print(f"✓ Reference saved to: {ref_save_path}")
			print(f"  (Use --reference_image {ref_save_path} for other cohorts)")
			
			# También guardar imagen visual de la referencia para inspección
			ref_vis_path = os.path.join(args.save_dir, 'macenko_reference.jpg')
			Image.fromarray(ref_region_rgb).save(ref_vis_path)
			print(f"✓ Reference visualization saved to: {ref_vis_path}")
		else:
			# Cargar referencia existente
			print(f"Loading reference from: {args.reference_image}")
			ref_image = np.load(args.reference_image)
			normalizer.fit(ref_image)
			print(f"✓ Normalizer fitted to reference")

	# ========================================================================
	# Ejecutar seg_and_patch con normalizer
	# ========================================================================

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip,
											normalizer=normalizer)
