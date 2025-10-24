
if [ "$1" != "cptac" ] && [ "$1" != "tcga" ]; then
    echo "Invalid database name. Use 'cptac' or 'tgca'."
    exit 1
else
    DATABASE_TRAIN=$1
    if [ "$1" != "cptac" ]; then
        DATABASE_TEST="cptac"
    else
        DATABASE_TEST="tcga"
    fi
fi

if [ "$2" == "pam50" ]; then
    CSV_FILE_TRAIN=data/dataset_csv/$DATABASE_TRAIN-subtype_pam50.csv
    CSV_FILE_TEST=data/dataset_csv/$DATABASE_TEST-subtype_pam50.csv
    LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
    CLAM_MODEL_TYPE=clam_mb
elif [ "$2" == "ihc" ]; then
    CSV_FILE_TRAIN=data/dataset_csv/$DATABASE_TRAIN-subtype_ihc.csv
    CSV_FILE_TEST=data/dataset_csv/$DATABASE_TEST-subtype_ihc.csv
    LABEL_DICT="{'Triple-negative':0,'Luminal-A':1,'Her2-not-luminal':2,'Luminal-B(HER2-)':3,'Luminal-B(HER2+)':4}"
    CLAM_MODEL_TYPE=clam_mb
elif [ "$2" == "ihc_simple" ]; then
    CSV_FILE_TRAIN=data/dataset_csv/$DATABASE_TRAIN-subtype_ihc_simple.csv
    CSV_FILE_TEST=data/dataset_csv/$DATABASE_TEST-subtype_ihc_simple.csv
    LABEL_DICT="{'Triple-negative':0,'Luminal-A':1,'Luminal-B':2,'HER2':3}"
    CLAM_MODEL_TYPE=clam_mb
elif [ "$2" == "erbb2" ]; then
    CSV_FILE_TRAIN=data/dataset_csv/$DATABASE_TRAIN-erbb2.csv
    CSV_FILE_TEST=data/dataset_csv/$DATABASE_TEST-erbb2.csv
    LABEL_DICT="{'negative':0,'positive':1}"
    CLAM_MODEL_TYPE=clam_sb
elif [ "$2" == "pr" ]; then
    CSV_FILE_TRAIN=data/dataset_csv/$DATABASE_TRAIN-pr.csv
    CSV_FILE_TEST=data/dataset_csv/$DATABASE_TEST-pr.csv
    LABEL_DICT="{'negative':0,'positive':1}"
    CLAM_MODEL_TYPE=clam_sb
elif [ "$2" == "er" ]; then
    CSV_FILE_TRAIN=data/dataset_csv/$DATABASE_TRAIN-er.csv
    CSV_FILE_TEST=data/dataset_csv/$DATABASE_TEST-er.csv
    LABEL_DICT="{'negative':0,'positive':1}"
    CLAM_MODEL_TYPE=clam_sb
else
    echo "Invalid parameter. Use 'ihc', 'ihc_simple', pam50', 'er', 'pr' or 'erbb2'."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Please provide the directory where the features are located regardless ."
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

if [ -z "$6" ] || [ "$6" -lt 1 ] || [ "$6" -gt 13 ]; then
    echo "Please provide a valid selection number for technique to test: [1,13]."
    exit 1
fi

SEED=42
K=1
DROP_OUT=0.7
LR=1e-4
REG=0.0001
BAG_LOSS=ce
INST_LOSS=ce
B=64
MODEL_SIZE=big
CUDA_DEV=0

LABEL_FRAC=1
VAL_FRAC=0.15
TEST_FRAC=0.0
CURRENT_TRAIN=$(date +"%s")
SPLIT_DIR_TRAIN=.splits/$DATABASE_TRAIN/ho-train-$CURRENT_TRAIN
python src/create_splits_seq.py --seed $SEED --k $K --test_frac $TEST_FRAC --val_frac $VAL_FRAC --split_dir $SPLIT_DIR_TRAIN --label_frac $LABEL_FRAC --csv_path $CSV_FILE_TRAIN --label_dict $LABEL_DICT $PATIENT_STRAT

LABEL_FRAC=1
VAL_FRAC=0.0
TEST_FRAC=0.0
CURRENT_TEST=$(date +"%s")
SPLIT_DIR_TEST=.splits/$DATABASE_TEST/ho-test-$CURRENT_TEST
python src/create_splits_seq.py --seed $SEED --k $K --test_frac $TEST_FRAC --val_frac $VAL_FRAC --split_dir $SPLIT_DIR_TEST --label_frac $LABEL_FRAC --csv_path $CSV_FILE_TEST --label_dict $LABEL_DICT $PATIENT_STRAT

case "$6" in
    1)
        echo "Parameter 6 is set to 1."
        EMBED_DIM=1024
        MODEL_NAME=cnn
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    2)
        echo "Parameter 6 is set to 2."
        EMBED_DIM=512
        MODEL_NAME=conch
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    3)
        echo "Parameter 6 is set to 3."
        EMBED_DIM=768
        MODEL_NAME=ctranspath
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    4)
        echo "Parameter 6 is set to 4."
        EMBED_DIM=768
        MODEL_NAME=hibou_b
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    5)
        echo "Parameter 6 is set to 5."
        EMBED_DIM=1024
        MODEL_NAME=hibou_l
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    6)
        echo "Parameter 6 is set to 6."
        EMBED_DIM=1536
        MODEL_NAME=hoptimus0
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    7)
        echo "Parameter 6 is set to 7."
        EMBED_DIM=2048
        MODEL_NAME=musk
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    8)
        echo "Parameter 6 is set to 8."
        EMBED_DIM=1024
        MODEL_NAME=phikon
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    9)
        echo "Parameter 6 is set to 9."
        EMBED_DIM=1536
        MODEL_NAME=provgigapath
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    10)
        echo "Parameter 6 is set to 10."
        EMBED_DIM=2048
        MODEL_NAME=retccl
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    11)
        echo "Parameter 6 is set to 11."
        EMBED_DIM=1024
        MODEL_NAME=uni
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    12)
        echo "Parameter 6 is set to 12."
        EMBED_DIM=1536
        MODEL_NAME=uni_2
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    13)
        echo "Parameter 6 is set to 13."
        EMBED_DIM=2560
        MODEL_NAME=virchow
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        RESULTS_DIR=.results/$DATABASE_TRAIN/$2/$CLAM_MODEL_TYPE/ho-$PATIENT_STRAT-$DATABASE_TRAIN-$CURRENT_TRAIN-$DATABASE_TEST-$CURRENT_TEST/$MODEL_NAME
        ;;
    *)
        echo "Parameter 6 is out of the valid range. Please provide a number between 1 and 13."
        exit 1
        ;;
esac

if [ -z "$5" ]; then
    echo "Please provide the fifth parameter as 'YES' or 'NO' to support HER2 virtualization (double dimension)."
    exit 1
elif [ "$5" == "YES" ]; then
    EMBED_DIM=$((EMBED_DIM * 2))
else
    EMBED_DIM=$EMBED_DIM
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/ho_main.py --B $B --reg $REG --model_size $MODEL_SIZE --seed $SEED --drop_out $DROP_OUT --early_stopping --lr $LR --k $K   --bag_loss $BAG_LOSS  --inst_loss $INST_LOSS --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR  --log_data --subtyping --data_root_dir_train $FEATURES_DIRECTORY_TRAIN --data_root_dir_test $FEATURES_DIRECTORY_TEST --embed_dim $EMBED_DIM --split_dir_train $SPLIT_DIR_TRAIN --split_dir_test $SPLIT_DIR_TEST --csv_path_train $CSV_FILE_TRAIN --csv_path_test $CSV_FILE_TEST --label_dict $LABEL_DICT --weighted_sample


