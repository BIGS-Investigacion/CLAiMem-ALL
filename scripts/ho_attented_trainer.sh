#!/bin/bash

# Script para entrenar CLAM con features agregados por atención
# Este script llama a ho_main_attended.py que integra toda la lógica de:
#   - Agregación de features por atención
#   - Creación de splits train/val/test
#   - Entrenamiento con CLAM
#
# Uso: bash scripts/ho_attented_trainer.sh <database> <subtype> <features_base_dir> <patient_strat> <her2_virt> <technique> <diversity> <top_k> <selection_method> <n_splits>

if [ "$1" != "cptac" ] && [ "$1" != "tcga" ]; then
    echo "Invalid database name. Use 'cptac' or 'tcga'."
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
    echo "Invalid parameter. Use 'ihc', 'ihc_simple', 'pam50', 'er', 'pr' or 'erbb2'."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Please provide the base directory where the features are located."
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

if [ -z "$7" ]; then
    echo "Please provide the seventh parameter as 'YES' or 'NO' to select extra diversity or not."
    exit 1
elif [ "$7" == "YES" ]; then
    DIVERSITY="--topo"
else
    DIVERSITY=""
fi

if [ -z "$8" ]; then
    echo "Using default top_k=100"
    TOP_K=100
else
    TOP_K=$8
fi

if [ -z "$9" ]; then
    echo "Using default selection_method=self_attention"
    SELECTION_METHOD="self_attention"
else
    SELECTION_METHOD=$9
fi

# Validar método de selección
if [ "$SELECTION_METHOD" != "attention" ] && [ "$SELECTION_METHOD" != "self_attention" ] && [ "$SELECTION_METHOD" != "norm" ] && [ "$SELECTION_METHOD" != "variance" ] && [ "$SELECTION_METHOD" != "random" ]; then
    echo "Invalid selection method. Use 'attention', 'self_attention', 'norm', 'variance', or 'random'."
    exit 1
fi

if [ -z "${10}" ]; then
    echo "Using default n_splits=10"
    N_SPLITS=10
else
    N_SPLITS=${10}
fi

SEED=42
K=1
DROP_OUT=0.3
LR=1e-4
REG=0.0001
BAG_LOSS=ce
INST_LOSS=ce
B=128
MODEL_SIZE=big
CUDA_DEV=0

# Determinar modelo y dimensiones según técnica
case "$6" in
    1)
        echo "Parameter 6 is set to 1."
        EMBED_DIM=1024
        MODEL_NAME=cnn
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    2)
        echo "Parameter 6 is set to 2."
        EMBED_DIM=512
        MODEL_NAME=conch
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    3)
        echo "Parameter 6 is set to 3."
        EMBED_DIM=768
        MODEL_NAME=ctranspath
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    4)
        echo "Parameter 6 is set to 4."
        EMBED_DIM=768
        MODEL_NAME=hibou_b
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    5)
        echo "Parameter 6 is set to 5."
        EMBED_DIM=1024
        MODEL_NAME=hibou_l
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    6)
        echo "Parameter 6 is set to 6."
        EMBED_DIM=1536
        MODEL_NAME=hoptimus0
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    7)
        echo "Parameter 6 is set to 7."
        EMBED_DIM=2048
        MODEL_NAME=musk
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    8)
        echo "Parameter 6 is set to 8."
        EMBED_DIM=1024
        MODEL_NAME=phikon
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    9)
        echo "Parameter 6 is set to 9."
        EMBED_DIM=1536
        MODEL_NAME=provgigapath
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    10)
        echo "Parameter 6 is set to 10."
        EMBED_DIM=2048
        MODEL_NAME=retccl
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    11)
        echo "Parameter 6 is set to 11."
        EMBED_DIM=1024
        MODEL_NAME=uni
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    12)
        echo "Parameter 6 is set to 12."
        EMBED_DIM=1536
        MODEL_NAME=uni_2
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
        ;;
    13)
        echo "Parameter 6 is set to 13."
        EMBED_DIM=2560
        MODEL_NAME=virchow
        EXP_CODE=$MODEL_NAME
        FEATURES_DIRECTORY_TRAIN_ORIGINAL=$F_DIRECTORY/$DATABASE_TRAIN/features_$MODEL_NAME
        FEATURES_DIRECTORY_TEST_ORIGINAL=$F_DIRECTORY/$DATABASE_TEST/features_$MODEL_NAME
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

# TEST: originales sin agregar
FEATURES_DIRECTORY_TEST=$FEATURES_DIRECTORY_TEST_ORIGINAL

# Llamar al nuevo script ho_main_attended.py que integra toda la lógica
CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/ho_main_attended.py \
    --database_train $DATABASE_TRAIN \
    --database_test $DATABASE_TEST \
    --subtype $2 \
    --top_k $TOP_K \
    --selection_method $SELECTION_METHOD \
    --n_splits $N_SPLITS \
    --data_root_dir_train_original $FEATURES_DIRECTORY_TRAIN_ORIGINAL \
    --data_root_dir_test $FEATURES_DIRECTORY_TEST \
    --csv_path_train $CSV_FILE_TRAIN \
    --csv_path_test $CSV_FILE_TEST \
    --label_dict "$LABEL_DICT" \
    --model_type $CLAM_MODEL_TYPE \
    --exp_code $EXP_CODE \
    --embed_dim $EMBED_DIM \
    --B $B \
    --reg $REG \
    --model_size $MODEL_SIZE \
    --seed $SEED \
    --drop_out $DROP_OUT \
    --early_stopping \
    --lr $LR \
    --k $K \
    --bag_loss $BAG_LOSS \
    --inst_loss $INST_LOSS \
    --log_data \
    --subtyping \
    $DIVERSITY \
    $(if [ -n "$PATIENT_STRAT" ]; then echo "--patient_strat"; fi)

echo ""
echo "================================================================================"
echo "TRAINING COMPLETED"
echo "================================================================================"
echo "Results saved to: $RESULTS_DIR"
echo "================================================================================"
