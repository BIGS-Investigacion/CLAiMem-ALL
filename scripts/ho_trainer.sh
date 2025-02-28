SEED=42
K=1
DROP_OUT=0.25
LR=2e-4
BAG_LOSS=ce
INST_LOSS=svm
CLAM_MODEL_TYPE=clam_sb
MODEL_SIZE=big
CUDA_DEV=0

DATABASE_TRAIN=cptac-brca
CSV_FILE_TRAIN=data/dataset_csv/brca-subtype_pam50.csv
LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
LABEL_FRAC=1
VAL_FRAC=0.1
TEST_FRAC=0.1
SPLIT_DIR_TRAIN=.splits/$DATABASE_TRAIN/ho_$K
python src/create_splits_seq.py --seed $SEED --k $K --test_frac $TEST_FRAC --val_frac $VAL_FRAC --split_dir $SPLIT_DIR_TRAIN --label_frac $LABEL_FRAC --csv_path $CSV_FILE_TRAIN --label_dict $LABEL_DICT ##--patient_strat

DATABASE_TEST=tcga-brca
CSV_FILE_TEST=data/dataset_csv/tcga-subtype_pam50.csv
LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
LABEL_FRAC=1
VAL_FRAC=0.1
TEST_FRAC=0.1
SPLIT_DIR_TEST=.splits/$DATABASE_TEST/ho_$K
python src/create_splits_seq.py --seed $SEED --k $K --test_frac $TEST_FRAC --val_frac $VAL_FRAC --split_dir $SPLIT_DIR_TEST --label_frac $LABEL_FRAC --csv_path $CSV_FILE_TEST --label_dict $LABEL_DICT #--patient_strat

EMBED_DIM=1024
MODEL_NAME=cnn
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY_TRAIN=.features/$DATABASE_TRAIN/features_$MODEL_NAME
FEATURES_DIRECTORY_TEST=.features/$DATABASE_TEST/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE_TRAIN/$DATABASE_TEST/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/ho_main.py --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir_train $FEATURES_DIRECTORY_TRAIN --data_root_dir_test $FEATURES_DIRECTORY_TEST --embed_dim $EMBED_DIM --split_dir_train $SPLIT_DIR_TRAIN --split_dir_test $SPLIT_DIR_TEST --csv_path_train $CSV_FILE_TRAIN --csv_path_test $CSV_FILE_TEST --label_dict $LABEL_DICT  