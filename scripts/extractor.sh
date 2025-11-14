#!/bin/bash
#Example usage: bash scripts/extractor.sh  tcga|cptac /path/to/data/ path/to/class_info.csv 0-13 0-1 0-1 macenko_file 
DATABASE=$1
DATA_DIRECTORY=$2
CSV_FILE_NAME=$3
PATCHES_DIRECTORY=.patches_40x/$DATABASE
PATCH_SIZE=512 # TO BE FIXED: Should be 128 if 10x, 256 if 20x, 512 if 40x...
FEATURES_BASE=.features_40x
CUDA_DEV=0
SLIDE_EXT=.svs

if [ "$4" -eq 0 ]; then
    PRESET_CSV=tcga.csv
    MACENKO=""
    if [ "$6" -eq 1 ]; then
        MACENKO="--use_macenko"
        if [ -n "$7" ]; then
            MACENKO="$MACENKO --reference_image $7"
            echo "Using macenko file: $7"
        fi
    fi
    python src/create_patches_fp.py --source $DATA_DIRECTORY --save_dir $PATCHES_DIRECTORY --patch_size $PATCH_SIZE --preset $PRESET_CSV --seg --patch --stitch $MACENKO
    python src/bigs_auxiliar/downloader.py
    
elif [ "$4" -eq 1 ]; then

    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_cnn
    MODEL_NAME=resnet50_trunc
    
elif [ "$4" -eq 2 ]; then
    
    export CONCH_CKPT_PATH='.checkpoint/conch/pytorch_model.bin'
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_conch
    MODEL_NAME=conch_v1
    
elif [ "$4" -eq 3 ]; then
    #Download pretrained model  ctranspath.pth from https://github.com/Xiyue-Wang/TransPath.git
    export GENERIC_CKPT_PATH=.checkpoint/ctranspath/ctranspath.pth
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_ctranspath
    MODEL_NAME=ctranspath
    

elif [ "$4" -eq 4 ]; then
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_hibou_b
    MODEL_NAME=hibou_b
    
elif [ "$4" -eq 5 ]; then
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_hibou_l
    MODEL_NAME=hibou_l
    

elif [ "$4" -eq 6 ]; then
    BATCH_SIZE=128
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_hoptimus0
    MODEL_NAME=hoptimus0
    

elif [ "$4" -eq 7 ]; then
    BATCH_SIZE=128
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_musk
    MODEL_NAME=musk
    
elif [ "$4" -eq 8 ]; then
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_phikon
    MODEL_NAME=phikon

elif [ "$4" -eq 9 ]; then
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_provgigapath
    MODEL_NAME=provgigapath

elif [ "$4" -eq 10 ]; then
    #Download pretrained model best_ckpt.pth from https://github.com/Xiyue-Wang/RetCCL
    export GENERIC_CKPT_PATH='.checkpoint/retccl/best_ckpt.pth'
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_retccl
    MODEL_NAME=retccl

elif [ "$4" -eq 11 ]; then
    export UNI_CKPT_PATH='.checkpoint/uni/pytorch_model.bin'
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_uni
    MODEL_NAME=uni_v1

elif [ "$4" -eq 12 ]; then
    export UNI_CKPT_PATH='.checkpoint/uni_2/pytorch_model.bin'
    BATCH_SIZE=128
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_uni_2
    MODEL_NAME=uni_v2
    CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $PATCHES_DIRECTORY --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name $MODEL_NAME

elif [ "$4" -eq 13 ]; then
    BATCH_SIZE=512
    FEATURES_DIRECTORY=$FEATURES_BASE/$DATABASE/features_virchow
    MODEL_NAME=virchow

else
    echo "Invalid parameter. Use [0-13]"
    exit 1
fi
EXTRA_ARGS=""
if [ "$5" -eq 1 ]; then
    EXTRA_ARGS="--virtual"
    #BATCH_SIZE=$((BATCH_SIZE * 2))
    echo "Using virtual patches."
fi

if [ "$6" -eq 1 ] && [ -n "$7" ]; then
    PATCHES_DIRECTORY=$PATCHES_DIRECTORY"_macenko"
    FEATURES_DIRECTORY=$FEATURES_DIRECTORY"_macenko"
    EXTRA_ARGS="$EXTRA_ARGS --use_macenko --reference_image $7"
    echo "Using macenko file: $7"
fi

echo $EXTRA_ARGS

if [ "$4" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_DEV python src/extract_features_fp.py --data_h5_dir $PATCHES_DIRECTORY --data_slide_dir $DATA_DIRECTORY --csv_path $CSV_FILE_NAME --feat_dir $FEATURES_DIRECTORY --batch_size $BATCH_SIZE --slide_ext $SLIDE_EXT --model_name $MODEL_NAME $EXTRA_ARGS
fi