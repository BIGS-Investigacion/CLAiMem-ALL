DATABASE_TRAIN=cptac-brca
DATABASE_TEST=tcga-brca
CSV_FILE_TEST=data/dataset_csv/tcga-subtype_pam50.csv
LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
RESULTS_DIR=.results/$DATABASE_TRAIN/$DATABASE_TEST
CLAM_MODEL_TYPE=clam_sb
MODEL_SIZE=small
CUDA_DEV=0
K=1

EMBED_DIM=1024
MODEL_NAME=cnn
FEATURES_DIRECTORY=.features/$DATABASE_TEST/features_$MODEL_NAME
EXP_CODE=$MODEL_NAME
EXP_CODE_SAVE=$MODEL_NAME\_summary
DROP_OUT=0.5

CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py --model_size $MODEL_SIZE --k $K --drop_out $DROP_OUT --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --csv_path $CSV_FILE_TEST --label_dict $LABEL_DICT --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM

#CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml

EMBED_DIM=1536
MODEL_NAME=uni_2
FEATURES_DIRECTORY=.features/$DATABASE/features_$MODEL_NAME
EXP_CODE=$MODEL_NAME
EXP_CODE_SAVE=$MODEL_NAME\_summary

#CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py --model_size $MODEL_SIZE --k $K --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --csv_path $CSV_FILE --label_dict $LABEL_DICT --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM

#CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml