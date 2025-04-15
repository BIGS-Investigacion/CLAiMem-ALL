export CONCH_CKPT_PATH='.checkpoint/conch/pytorch_model.bin'
CUDA_VISIBLE_DEVICES=0 python src/create_heatmaps.py --config config/heatmaps/pam50.yaml