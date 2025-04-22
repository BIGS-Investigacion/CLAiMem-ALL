#!/bin/bash
if [ -z "$1" ]; then
    echo "Please provide the directory where the features are located."
    exit 1
else
    F_DIRECTORY=$1
fi

if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$2" -lt 0 ]; then
    echo "Please provide a non-negative integer for the cross-validation parameter. 0 means hold-out"
    exit 1
elif [ "$2" -eq 0 ]; then
    K=1
    VALIDATION="ho"
else
    K=$2
    VALIDATION=$2-"cv"
fi


SEED=42
DROP_OUT=0.7
LR=1e-4
REG=0.0001
BAG_LOSS=ce
INST_LOSS=ce
B=64
MODEL_SIZE=big
CUDA_DEV=0
RESULTS_DIR=.results

for dir_database in $RESULTS_DIR/*; do
    if [ -d "$dir_database" ]; then
        DATABASE=$(basename $dir_database)
        if [ "$K" -eq 1 ] && [ "$DATABASE" == 'cptac' ] ; then
            DATABASE=tcga
        elif [ "$K" -eq 1 ] && [ "$DATABASE" == 'tcga' ] ; then
            DATABASE=cptac    
        fi
        for aim in $dir_database/*; do
            if [[ $aim == *"pam50" ]]; then
                CSV_FILE=data/dataset_csv/$DATABASE-subtype_pam50.csv
                LABEL_DICT="{'basal':0,'her2':1,'luma':2,'lumb':3,'normal':4}"
                CLAM_MODEL_TYPE=clam_mb
            elif [[ $aim == *"ihc" ]]; then
                CSV_FILE=data/dataset_csv/$DATABASE-subtype_ihc.csv
                LABEL_DICT="{'Triple-negative':0,'Luminal-A':1,'Her2-not-luminal':2,'Luminal-B(HER2-)':3,'Luminal-B(HER2+)':4}"
                CLAM_MODEL_TYPE=clam_mb
            elif [[ $aim == *"ihc_simple" ]]; then
                CSV_FILE=data/dataset_csv/$DATABASE-subtype_ihc_simple.csv
                LABEL_DICT="{'Triple-negative':0,'Luminal-A':1,'Luminal-B':2,'HER2':3}"
                CLAM_MODEL_TYPE=clam_mb
            elif [[ $aim == *"erbb2" ]]; then
                CSV_FILE=data/dataset_csv/$DATABASE-erbb2.csv
                LABEL_DICT="{'negative':0,'positive':1}"
                CLAM_MODEL_TYPE=clam_sb
            elif [[ $aim == *"pr" ]]; then
                CSV_FILE=data/dataset_csv/$DATABASE-pr.csv
                LABEL_DICT="{'negative':0,'positive':1}"
                CLAM_MODEL_TYPE=clam_sb
            elif [[ $aim == *"er" ]]; then
                CSV_FILE=data/dataset_csv/$DATABASE-er.csv
                LABEL_DICT="{'negative':0,'positive':1}"
                CLAM_MODEL_TYPE=clam_sb
            else
                echo "Invalid parameter. Use 'ihc', 'ihc_simple', 'pam50', 'er', 'pr' or 'erbb2'."
                exit 1
            fi
            
            for test in $aim/$CLAM_MODEL_TYPE/$VALIDATION*/*; do
                valid=True
                if [[ $test == *"cnn"* ]]; then
                    EMBED_DIM=1024
                    MODEL_NAME=cnn
                elif [[ $test == *"conch"* ]]; then
                    EMBED_DIM=512
                    MODEL_NAME=conch
                elif [[ $test == *"ctranspath"* ]]; then
                    EMBED_DIM=768
                    MODEL_NAME=ctranspath
                elif [[ $test == *"hibou_b"* ]]; then
                    EMBED_DIM=768
                    MODEL_NAME=hibou_b
                elif [[ $test == *"hibou_l"* ]]; then
                    EMBED_DIM=1024
                    MODEL_NAME=hibou_l
                elif [[ $test == *"hoptimus0"* ]]; then
                    EMBED_DIM=1536
                    MODEL_NAME=hoptimus0
                elif [[ $test == *"provgigapath"* ]]; then
                    EMBED_DIM=1536
                    MODEL_NAME=provgigapath
                elif [[ $test == *"phikon"* ]]; then
                    EMBED_DIM=1024
                    MODEL_NAME=phikon
                elif [[ $test == *"uni"* ]]; then
                    EMBED_DIM=1024
                    MODEL_NAME=uni
                elif [[ $test == *"uni_2"* ]]; then
                    EMBED_DIM=1536
                    MODEL_NAME=uni_2
                elif [[ $test == *"musk"* ]]; then
                    EMBED_DIM=2048
                    MODEL_NAME=musk
                elif [[ $test == *"retccl"* ]]; then
                    EMBED_DIM=2048
                    MODEL_NAME=retccl
                elif [[ $test == *"virchow"* ]]; then
                    EMBED_DIM=2560
                    MODEL_NAME=virchow
                else
                    echo $test
                    echo "Invalid parameter. Use 'conch', 'ctranspath', 'hibou_b', 'hoptimus0', 'provgigapath', 'uni_2', 'musk', 'retccl' or 'virchow'."
                    valid=False
                fi
                if [ "$valid" = True ]; then
                    FEATURES_DIRECTORY=$F_DIRECTORY/$DATABASE/features_$MODEL_NAME
                    EXP_CODE=$MODEL_NAME
                    EXP_CODE_SAVE=$(echo $test | tr '/' '-')-$MODEL_NAME-summary
                    CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py  --drop_out $DROP_OUT --model_size $MODEL_SIZE --k $K --models_exp_code $EXP_CODE --save_exp_code $EXP_CODE_SAVE --csv_path $CSV_FILE --label_dict $LABEL_DICT --model_type $CLAM_MODEL_TYPE --results_dir $test --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM
                    #CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml
                fi
            done
        done
    fi
done







