python src/he2her2.py --data_root ../../datos/BCI_dataset --out_dir runs/he2her2_pix2pix \
    --image_size 1024 --batch_size 4 --epochs 200
python src/he2her2.py --mode infer --data_root ../../datos/BCI_dataset \
    --weights runs/he2her2_pix2pix/last.pt --out_dir runs/he2her2_pix2pix --image_size 1024
