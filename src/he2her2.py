import os
import glob
import argparse
from PIL import Image
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


# ----------------------------
# Utilidades
# ----------------------------
def list_common_filenames(he_dir: str, ihc_dir: str, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> List[str]:
    """Devuelve la intersección de nombres de archivo presentes en HE e IHC."""
    def names_in(d):
        files = []
        for e in exts:
            files += glob.glob(os.path.join(d, f"*{e}"))
        return set([os.path.basename(f) for f in files])
    he_names = names_in(he_dir)
    ihc_names = names_in(ihc_dir)
    commons = sorted(list(he_names & ihc_names))
    if len(commons) == 0:
        raise RuntimeError(f"No se encontraron nombres comunes entre:\n  {he_dir}\n  {ihc_dir}")
    return commons


class HE2IHCDataset(Dataset):
    """Pares alineados HE (entrada) -> IHC/HER2 (objetivo) emparejados por nombre."""

    def __init__(self, root: str, split: str = "train", image_size: int = 256, augment: bool = True):
        assert split in ["train", "test"]
        self.he_dir = os.path.join(root, "HE", split)
        self.ihc_dir = os.path.join(root, "IHC", split)
        self.filenames = list_common_filenames(self.he_dir, self.ihc_dir)

        tfs = [transforms.Resize((image_size, image_size), Image.BICUBIC)]
        if augment and split == "train":
            tfs += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=10),
            ]
        tfs += [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1]
        ]
        self.tf = transforms.Compose(tfs)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        name = self.filenames[idx]
        he = Image.open(os.path.join(self.he_dir, name)).convert("RGB")
        ihc = Image.open(os.path.join(self.ihc_dir, name)).convert("RGB")
        return self.tf(he), self.tf(ihc), name


# ----------------------------
# Modelos (pix2pix)
# ----------------------------
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x): return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.model(x)
        x = self.dropout(x)
        return torch.cat((x, skip), 1)


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        # Decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class Discriminator(nn.Module):
    """PatchGAN: concatena [HE, IHC] y produce mapa de real/fake."""
    def __init__(self, in_channels=3):
        super().__init__()
        def C(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *C(in_channels * 2, 64, norm=False),
            *C(64, 128),
            *C(128, 256),
            *C(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)  # tamaño de salida depende del input
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# ----------------------------
# Entrenamiento
# ----------------------------
def save_sample(he, fake, real, out_dir, name, max_batch_show=4):
    os.makedirs(out_dir, exist_ok=True)
    denorm = lambda t: (t + 1) * 0.5
    b = min(he.size(0), max_batch_show)
    grid = torch.cat([denorm(he[:b]), denorm(fake[:b]), denorm(real[:b])], dim=0)
    save_image(grid, os.path.join(out_dir, f"{name}_vis.png"), nrow=b)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Dispositivo: {device}")

    train_set = HE2IHCDataset(args.data_root, "train", args.image_size, augment=not args.no_aug)
    test_set  = HE2IHCDataset(args.data_root, "test", args.image_size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    G, D = GeneratorUNet().to(device), Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion_GAN, criterion_L1 = nn.MSELoss(), nn.L1Loss()

    start_epoch = 0
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "last.pt")

    if args.resume and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ckpt["G"]); D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"]); opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reanudado desde epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)
        for he, ihc, _ in pbar:
            he, ihc = he.to(device), ihc.to(device)

            # -----------------
            #  Generador
            # -----------------
            opt_G.zero_grad()
            fake_ihc = G(he)
            pred_fake_for_G = D(he, fake_ihc)

            # etiquetas dinámicas del tamaño correcto
            valid_for_G = torch.ones_like(pred_fake_for_G)

            loss_GAN = criterion_GAN(pred_fake_for_G, valid_for_G)
            loss_L1  = criterion_L1(fake_ihc, ihc) * args.lambda_l1
            loss_G   = loss_GAN + loss_L1
            loss_G.backward()
            opt_G.step()

            # -----------------
            #  Discriminador
            # -----------------
            opt_D.zero_grad()
            pred_real = D(he, ihc)
            valid_for_D = torch.ones_like(pred_real)
            loss_real = criterion_GAN(pred_real, valid_for_D)

            pred_fake = D(he, fake_ihc.detach())
            fake_for_D = torch.zeros_like(pred_fake)
            loss_fake = criterion_GAN(pred_fake, fake_for_D)

            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            opt_D.step()

            pbar.set_postfix({"loss_G": f"{loss_G.item():.3f}",
                              "loss_D": f"{loss_D.item():.3f}",
                              "L1": f"{loss_L1.item():.3f}"})

        # Muestras y checkpoint
        G.eval()
        with torch.no_grad():
            he_b, ihc_b, _ = next(iter(test_loader))
            he_b, ihc_b = he_b.to(device), ihc_b.to(device)
            fake_b = G(he_b)
            save_sample(he_b, fake_b, ihc_b, os.path.join(args.out_dir, "samples"), f"epoch_{epoch+1:03d}")

        torch.save({
            "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "args": vars(args),
        }, ckpt_path)

    torch.save({"G": G.state_dict()}, os.path.join(args.out_dir, "G_final.pt"))
    print("Entrenamiento terminado.")


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Dispositivo: {device}")

    G = GeneratorUNet().to(device).eval()
    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["G"] if "G" in ckpt else ckpt
    G.load_state_dict(state)

    test_set = HE2IHCDataset(args.data_root, "test", args.image_size, augment=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    out_dir = os.path.join(args.out_dir, "inference")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for he, ihc, names in tqdm(test_loader, desc="Inferencia"):
            # opcional: pásalos TODOS a GPU
            he = he.to(device)
            ihc = ihc.to(device)

            fake = G(he)

            for i in range(fake.size(0)):
                base, _ = os.path.splitext(names[i])

                # Guardar solo la predicción
                save_image((fake[i].detach().cpu() + 1) * 0.5,
                           os.path.join(out_dir, f"{base}_pred.png"))

                # Guardar triptico HE | Pred | GT
                he_cpu   = he[i].detach().cpu()
                fake_cpu = fake[i].detach().cpu()
                ihc_cpu  = ihc[i].detach().cpu()
                triplet = torch.stack([he_cpu, fake_cpu, ihc_cpu], dim=0)
                save_image((triplet + 1) * 0.5,
                           os.path.join(out_dir, f"{base}_triplet.png"),
                           nrow=3)

def parse_args():
    p = argparse.ArgumentParser(description="HE → HER2 (IHC) con pix2pix")
    p.add_argument("--data_root", type=str, default="BCI_dataset", help="Ruta al dataset raíz")
    p.add_argument("--out_dir", type=str, default="runs/he2her2_pix2pix", help="Directorio de salidas")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--no_aug", action="store_true", help="Desactivar augmentación")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU")
    p.add_argument("--resume", action="store_true", help="Reanudar si existe runs/last.pt")
    p.add_argument("--mode", choices=["train", "infer"], default="train")
    p.add_argument("--weights", type=str, default="", help="Ruta a pesos para inferencia")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "train":
        train(args)
    else:
        if not args.weights:
            raise ValueError("--weights es obligatorio en modo infer")
        infer(args)
