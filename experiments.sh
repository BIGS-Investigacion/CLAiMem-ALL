#source route_to_conda/bin/activate
#conda env create -f env.yml
#conda activate clam_latest

#DATA_DIRECTORY=/media/jorge/Expansion/medicina/patologia_digital/datos/histology/perfil_molecular/publicas/TCGA-BRCA/diagnostico
DATA_ROOT_DIR=data
DATA_DIRECTORY=$DATA_ROOT_DIR/original/tcga_sample
RESULT_DIRECTORY=$DATA_ROOT_DIR/processed/tcga_sample
PATCH_SIZE=256
PRESET_CSV=tcga.csv

#python src/create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULT_DIRECTORY --patch_size $PATCH_SIZE --preset $PRESET_CSV --seg --patch --stitch

CSV_FILE_NAME=$DATA_ROOT_DIR/dataset_csv/tcga-subtype_short.csv
DIR_TO_COORDS=$RESULT_DIRECTORY

BATCH_SIZE=512
SLIDE_EXT=.svs
CUDA_DEV=0

#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_cnn
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT

#export CONCH_CKPT_PATH=checkpoint/conch/pytorch_model.bin
#BATCH_SIZE=512
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_conch
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name conch_v1

#export UNI_CKPT_PATH=checkpoint/uni/pytorch_model.bin
#BATCH_SIZE=500
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_uni
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name uni_v1

#export UNI_CKPT_PATH=checkpoint/uni_2/pytorch_model.bin
#BATCH_SIZE=128
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_uni_2
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name uni_v2

#Download pretrained model  ctranspath.pth from https://github.com/Xiyue-Wang/TransPath.git
#export GENERIC_CKPT_PATH=checkpoint/ctranspath/ctranspath.pth
#BATCH_SIZE=256
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_ctranspath
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name ctranspath

#Download pretrained model best_ckpt.pth from https://github.com/Xiyue-Wang/RetCCL
#export GENERIC_CKPT_PATH=checkpoint/retccl/best_ckpt.pth
#BATCH_SIZE=512
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_retccl
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name retccl

BATCH_SIZE=128
FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_provgigapath
CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name provgigapath
