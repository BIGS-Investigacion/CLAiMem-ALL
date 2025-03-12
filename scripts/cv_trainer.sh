
if [ "$1" != "cptac" ] && [ "$1" != "tcga" ]; then
    echo "Invalid database name. Use 'cptac' or 'tgca'."
    exit 1
else
    DATABASE=$1
fi

if [ "$2" == "pam50" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-subtype_pam50.csv
    LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
elif [ "$2" == "erbb2" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-erbb2.csv
    LABEL_DICT="{'negative':0,'positive':1}"
elif [ "$2" == "pr" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-pr.csv
    LABEL_DICT="{'negative':0,'positive':1}"
elif [ "$2" == "er" ]; then
    CSV_FILE=data/dataset_csv/$DATABASE-er.csv
    LABEL_DICT="{'negative':0,'positive':1}"
else
    echo "Invalid parameter. Use 'pam50', 'er', 'pr' or 'erbb2'."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Please provide the directory where the features are located."
    exit 1
else
    F_DIRECTORY=$3
fi

if [ -z "$4" ]; then
    echo "Please provide the fourth parameter as 'YES' or 'NO' to select patient stratification or not."
    exit 1
elif [ "$4" == "YES" ]; then
    PATIENT_STRAT="--patient_strat"
else
    PATIENT_STRAT=""
fi

SEED=42
K=2
DROP_OUT=0.5
LR=2e-4
BAG_LOSS=ce
INST_LOSS=svm
CLAM_MODEL_TYPE=clam_sb
B=32
MODEL_SIZE=big
LABEL_FRAC=1
TEST_FRAC=0.1
VAL_FRAC=0.2
CURRENT=$(date +"%s")
SPLIT_DIR=.splits/$DATABASE/cv-$K-$CURRENT
CUDA_DEV=0

python src/create_splits_seq.py --seed $SEED --k $K --test_frac $TEST_FRAC --val_frac $VAL_FRAC --split_dir $SPLIT_DIR --label_frac $LABEL_FRAC --csv_path $CSV_FILE --label_dict $LABEL_DICT $PATIENT_STRAT

EMBED_DIM=1024
MODEL_NAME=cnn
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=512
MODEL_NAME=conch
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=ctranspath
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=hibou_b
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=hibou_l
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=hoptimus0
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=musk
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=phikon
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=provgigapath
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=retccl
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=uni
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  


EMBED_DIM=1536
MODEL_NAME=uni_2
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  

EMBED_DIM=1024
MODEL_NAME=virchow
EXP_CODE=$MODEL_NAME
FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
RESULTS_DIR=.results/$DATABASE/$2/$CLAM_MODEL_TYPE/$K-cv-$PATIENT_STRAT-$CURRENT/$MODEL_NAME

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/cv_main.py --B $B --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --weighted_sample --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR --csv_path $CSV_FILE --label_dict $LABEL_DICT  
