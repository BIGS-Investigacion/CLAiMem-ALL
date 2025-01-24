#source route_to_conda/bin/activate
#conda env create -f env.yml
#conda activate clam_latest

#DATA_DIRECTORY=/media/jorge/Expansion/medicina/patologia_digital/datos/histology/perfil_molecular/publicas/TCGA-BRCA/diagnostico
DATA_ROOT_DIR=data
DATA_DIRECTORY=$DATA_ROOT_DIR/original/tcga_sample
RESULT_DIRECTORY=$DATA_ROOT_DIR/tcga_sample_results
PATCH_SIZE=256
PRESET_CSV=tcga.csv

python src/create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULT_DIRECTORY --patch_size $PATCH_SIZE --preset $PRESET_CSV --seg --patch --stitch

DIR_TO_COORDS=$RESULT_DIRECTORY/patches
CSV_FILE_NAME=$DATA_ROOT_DIR/dataset_csv/brca-subtype.csv
FEATURES_DIRECTORY=$RESULT_DIRECTORY/features
BATCH_SIZE=512
SLIDE_EXT=.svs
CUDA_DEV=0
CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT
