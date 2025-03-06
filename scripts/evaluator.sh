#!/bin/bash

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

if ! [[ "$4" =~ ^[0-9]+$ ]] || [ "$4" -lt 0 ]; then
    echo "Please provide a non-negative integer for the cross-validation parameter. 0 means hold-out"
    exit 1
elif [ "$4" -eq 0 ]; then
    K=1
    VALIDATION="ho"
else
    K=$4
    VALIDATION=$4-"cv"
fi

CLAM_MODEL_TYPE=clam_sb
MODEL_SIZE=big
DROP_OUT=0.5
CUDA_DEV=0

for dir in .results/$DATABASE/$2/$CLAM_MODEL_TYPE/$VALIDATION*; do
    if [ -d "$dir" ]; then
        RESULTS_DIR=$dir
        
        for subdir in $RESULTS_DIR/*; do
            if [ -d "$subdir" ]; then
                MODEL_NAME=$(basename "$subdir")
                # Add your processing commands here
                if [[ "$MODEL_NAME" == "conch" ]]; then
                    EMBED_DIM=512
                else
                    EMBED_DIM=1024
                fi
                FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
                EXP_CODE=$MODEL_NAME
                EXP_CODE_SAVE=$(echo $dir | tr '/' '-')\_summary
                CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py  --drop_out $DROP_OUT --model_size $MODEL_SIZE --k $K --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --csv_path $CSV_FILE --label_dict $LABEL_DICT --model_type $CLAM_MODEL_TYPE --results_dir $RESULTS_DIR --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM
                #CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml
            fi
        done
    fi
done




