#source route_to_conda/bin/activate
#conda env create -f env.yml
#conda activate clam_latest


#DATA_DIRECTORY=/media/jorge/SP_PHD_U3/perfil_molecular/publicas/CPTAC-BRCA/BRCA
DATA_DIRECTORY=/media/jorge/Expansion/medicina/patologia_digital/datos/histology/clasificacion_cancer/perfil_molecular/publicas/CPTAC-BRCA/BRCA
DATA_ROOT_DIR=data
RESULT_DIRECTORY=$DATA_ROOT_DIR/processed/brca_sample
PATCH_SIZE=256
PRESET_CSV=tcga.csv

#python src/create_patches_fp.py --source $DATA_DIRECTORY --save_dir $RESULT_DIRECTORY --patch_size $PATCH_SIZE --preset $PRESET_CSV --seg --patch --stitch

CSV_FILE_NAME=$DATA_ROOT_DIR/dataset_csv/brca-subtype.csv
DIR_TO_COORDS=$RESULT_DIRECTORY

BATCH_SIZE=512
SLIDE_EXT=.svs
CUDA_DEV=0


python src/bigs_auxiliar/downloader.py
BATCH_SIZE=256
FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_cnn

#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT

K=10
SEED=42

python src/create_splits_seq.py --task task_4_brca_breast_mollecular_subtyping --seed $SEED --k $K

DROP_OUT=0.25
LR=2e-4
EXP_CODE=brca_breast_mollecular_subtyping_resnet
SPLIT_DIR=task_4_brca_breast_mollecular_subtyping_100
DATA_ROOT_DIR=$FEATURES_DIRECTORY
CUDA_VISIBLE_DEVICES=0 python src/main.py --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --exp_code $EXP_CODE --weighted_sample --bag_loss ce --inst_loss svm --task task_4_brca_breast_mollecular_subtyping --model_type clam_sb --log_data --subtyping --data_root_dir $DATA_ROOT_DIR --embed_dim 1024 --split_dir $SPLIT_DIR   

#CUDA_VISIBLE_DEVICES=0 python eval.py --k 10 --models_exp_code task_4_brca_breast_mollecular_subtyping --save_exp_code task_4_brca_breast_mollecular_subtyping_CLAM_10_s1_cv --task task_4_brca_breast_mollecular_subtyping --model_type clam_sb --results_dir results --data_root_dir DATA_ROOT_DIR --embed_dim 1024

#---------------------------------------OTHER MODELS---------------------------------------

#Download pretrained model  ctranspath.pth from https://github.com/Xiyue-Wang/TransPath.git
#export GENERIC_CKPT_PATH=checkpoint/ctranspath/ctranspath.pth
#BATCH_SIZE=256
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_ctranspath
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name ctranspath

#Download pretrained model best_ckpt.pth from https://github.com/Xiyue-Wang/RetCCL
#export GENERIC_CKPT_PATH=checkpoint/retccl/best_ckpt.pth
#BATCH_SIZE=256
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_retccl
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name retccl

#export CONCH_CKPT_PATH=checkpoint/conch/pytorch_model.bin
#BATCH_SIZE=256
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_conch
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name conch_v1

#export UNI_CKPT_PATH=checkpoint/uni/pytorch_model.bin
#BATCH_SIZE=256
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_uni
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name uni_v1

#BATCH_SIZE=64
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_provgigapath
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name provgigapath

#BATCH_SIZE=512
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_hibou_b
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name hibou_b

#BATCH_SIZE=512
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_hibou_l
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name hibou_l

#BATCH_SIZE=64
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_hoptimus0
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name hoptimus0

#BATCH_SIZE=512
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_virchow
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name virchow

#BATCH_SIZE=256
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_phikon
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name phikon

#BATCH_SIZE=64
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_musk
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name musk

#export UNI_CKPT_PATH=checkpoint/uni_2/pytorch_model.bin
#BATCH_SIZE=128
#FEATURES_DIRECTORY=$RESULT_DIRECTORY/features_uni_2
#CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $DIR_TO_COORDS --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name uni_v2

