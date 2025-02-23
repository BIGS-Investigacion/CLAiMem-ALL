DATABASE=cptac-brca
CSV_FILE=data/dataset_csv/brca-subtype_4c.csv
LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3}"
SEED=42
K=10
DROP_OUT=0.25
LR=2e-4
BAG_LOSS=ce
INST_LOSS=svm
CLAM_MODEL_TYPE=clam_sb
RESULT_DIRECTORY=.results/$DATABASE
LABEL_FRAC=1
SPLIT_DIR=.splits/$DATABASE/cv_$K
CUDA_DEV=0

python src/create_splits_seq.py --seed $SEED --k $K --split_dir $SPLIT_DIR --label_frac $LABEL_FRAC --csv_path $CSV_FILE --label_dict $LABEL_DICT

EMBED_DIM=1024
MODEL_NAME=cnn
EXP_CODE=brca_breast_mollecular_subtyping_$MODEL_NAME
FEATURES_DIRECTORY=.features/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/main.py --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --exp_code $EXP_CODE --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

#CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py --k $K --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --task $TASK --model_type $CLAM_MODEL_TYPE --results_dir $RESULT_DIRECTORY --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM

#CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml

EMBED_DIM=1536
MODEL_NAME=uni_2
EXP_CODE=brca_breast_mollecular_subtyping_$MODEL_NAME
FEATURES_DIRECTORY=.features/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/main.py --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --exp_code $EXP_CODE --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

#CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py --k $K --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --task $TASK --model_type $CLAM_MODEL_TYPE --results_dir $RESULT_DIRECTORY --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM

#CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml