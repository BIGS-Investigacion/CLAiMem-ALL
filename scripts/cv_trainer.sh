DATABASE=cptac-brca
#CSV_FILE=data/dataset_csv/brca-subtype_pam50.csv
#LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
CSV_FILE=data/dataset_csv/brca-erbb2.csv
LABEL_DICT="{'negative':0,'positive':1}"
SEED=42
K=1
DROP_OUT=0.5
LR=2e-4
BAG_LOSS=ce
INST_LOSS=svm
CLAM_MODEL_TYPE=clam_mb
MODEL_SIZE=big
LABEL_FRAC=1
TEST_FRAC=0.1
VAL_FRAC=0.1
SPLIT_DIR=.splits/$DATABASE/cv_$K
CUDA_DEV=0


python src/create_splits_seq.py --seed $SEED --k $K --test_frac $TEST_FRAC --val_frac $VAL_FRAC --split_dir $SPLIT_DIR --label_frac $LABEL_FRAC --csv_path $CSV_FILE --label_dict $LABEL_DICT #--force_balance

EMBED_DIM=1024
MODEL_NAME=cnn
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=.features/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/main.py --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1536
MODEL_NAME=uni_2
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=.features/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/main.py --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  