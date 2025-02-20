K=1
SEED=42
TASK=task_4_brca_breast_mollecular_subtyping

python src/create_splits_seq.py --task $TASK --seed $SEED --k $K

SPLIT_DIR=task_4_brca_breast_mollecular_subtyping_100
DROP_OUT=0.25
LR=2e-4
EMBED_DIM=1024
RESULT_DIRECTORY=.results
EXP_CODE=brca_breast_mollecular_subtyping_
MODELS_EXP_CODE=$EXP_CODE\_s1
EXP_CODE_SAVE=$MODELS_EXP_CODE\_cv_sum_up

BAG_LOSS=ce
INST_LOSS=svm
CLAM_MODEL_TYPE=clam_sb

#CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/main.py --drop_out $DROP_OUT --early_stopping --lr $LR --k $K --exp_code $EXP_CODE --weighted_sample --bag_loss $BAG_LOSS --inst_loss $INST_LOSS --task $TASK --model_type $CLAM_MODEL_TYPE --results_dir $RESULT_DIRECTORY --log_data --subtyping --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM --split_dir $SPLIT_DIR

#CUDA_VISIBLE_DEVICES=$CUDA_DEV  python src/eval.py --k $K --models_exp_code $MODELS_EXP_CODE --save_exp_code $EXP_CODE_SAVE --task $TASK --model_type clam_sb --results_dir $RESULT_DIRECTORY --data_root_dir $FEATURES_DIRECTORY --embed_dim $EMBED_DIM

#CUDA_VISIBLE_DEVICES=0 src/python create_heatmaps.py --config config_template.yaml

