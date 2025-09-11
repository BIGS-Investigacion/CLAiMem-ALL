python src/he2her2.py --data_root ../../datos/BCI_dataset --out_dir runs/he2her2_pix2pix \
    --image_size 512 --batch_size 32 --epochs 200 --resume 
python src/he2her2.py --mode infer --data_root ../../datos/BCI_dataset \
    --weights runs/he2her2_pix2pix/G_final.pt --out_dir runs/he2her2_pix2pix --image_size 512
