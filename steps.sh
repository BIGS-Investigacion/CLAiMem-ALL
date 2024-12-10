#source route_to_conda/bin/activate
#conda env create -f env.yml
conda activate clam_latest

#SOURCE=/media/jorge/Expansion/medicina/patologia_digital/datos/histology/perfil_molecular/publicas/TCGA-BRCA/diagnostico
DATA_DIRECTORY=/media/jorge/investigacion/datos/breast
RESULT_DIRECTORY=./results
DIR_TO_COORDS=$RESULT_DIRECTORY/patches
PATCH_SIZE=256
PRESET_CSV=tcga.csv
CSV_FILE_NAME=$DATA_DIRECTORY/trainer.csv
FEATURES_DIRECTORY=$RESULT_DIRECTORY/features
BATCH_SIZE=512
SLIDE_EXT=.svs
python create_patches_fp.py --source $SOURCE --save_dir $RESULT_DIRECTORY --patch_size $PATCH_SIZE --preset $PRESET_CSV --seg --patch --stitch
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT
