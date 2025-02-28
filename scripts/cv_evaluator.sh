DATABASE=cptac-brca
CSV_FILE=data/dataset_csv/brca-subtype_pam50.csv
LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
RESULTS_DIR=.results/$DATABASE
CLAM_MODEL_TYPE=clam_mb
MODEL_SIZE=big
CUDA_DEV=0
K=1

EMBED_DIM=1024
MODEL_NAME=cnn
FEATURES_DIRECTORY=.features/$DATABASE/features_$MODEL_NAME
EXP_CODE=$MODEL_NAME
EXP_CODE_SAVE=$MODEL_NAME\_summary

CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py --model_size $MODEL_SIZE --k $K --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --csv_path $CSV_FILE --label_dict $LABEL_DICT --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM

#CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml

EMBED_DIM=1536
MODEL_NAME=uni_2
FEATURES_DIRECTORY=.features/$DATABASE/features_$MODEL_NAME
EXP_CODE=$MODEL_NAME
EXP_CODE_SAVE=$MODEL_NAME\_summary

CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py --model_size $MODEL_SIZE --k $K --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --csv_path $CSV_FILE --label_dict $LABEL_DICT --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM

#CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml