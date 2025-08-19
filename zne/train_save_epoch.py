import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
import torchvision.models as models
from tqdm import tqdm
import pennylane as qml
from pennylane import numpy as np
import pennylane.qnn as qnn
import random

# ----- Repro -----
SEED = 123
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything(SEED)
torch.set_default_dtype(torch.float32)

# ---------- Dataset ----------
class EuroSATMultimodalDataset(Dataset):
    def __init__(self, csv_file, rgb_root, tif_root, transform_rgb=None, transform_tif=None):
        self.data = pd.read_csv(csv_file)
        self.rgb_root, self.tif_root = rgb_root, tif_root
        self.transform_rgb, self.transform_tif = transform_rgb, transform_tif
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rgb_path = os.path.join(self.rgb_root, row['Filename'])
        tif_path = os.path.join(self.tif_root, row['Filename'].replace('.jpg', '.tif'))
        rgb_image = Image.open(rgb_path).convert("RGB")
        if self.transform_rgb: rgb_image = self.transform_rgb(rgb_image)
        with rasterio.open(tif_path) as tif: tif_image = tif.read()
        tif_image = torch.tensor(tif_image/10000.0, dtype=torch.float32)
        if self.transform_tif: tif_image = self.transform_tif(tif_image)
        return rgb_image, tif_image, int(row['Label'])

# ---------- Transforms / Paths ----------
transform_rgb = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
class ResizeTIF:
    def __init__(self, size=(224,224)): self.size=size
    def __call__(self,x): return torch.nn.functional.interpolate(x.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
transform_tif = ResizeTIF((224,224))

base = os.path.expanduser("~/quantum/jsac/EuroSAT_extracted")
rgb_root = base
tif_root = os.path.expanduser("~/quantum/jsac/EuroSAT_extracted/allBands")
csv_train = os.path.join(base,"train.csv")
csv_val   = os.path.join(base,"validation.csv")

train_ds = EuroSATMultimodalDataset(csv_train, rgb_root, tif_root, transform_rgb, transform_tif)
val_ds   = EuroSATMultimodalDataset(csv_val,   rgb_root, tif_root, transform_rgb, transform_tif)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Classical feature extractors ----------
class SSRAN3(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self,x): return self.backbone(x).view(x.size(0),-1)

class LViT4E(nn.Module):
    def __init__(self, input_channels=13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(256,512)
    def forward(self,x): return self.fc(self.encoder(x).view(x.size(0),-1))

# ---------- Quantum layer (train at λ=1) ----------
noise_factor_base = 0.05      # ↓↓↓ milder base noise (was 0.1)
LAMBDA_SCALE = 1.0            # train once at λ=1
SHOTS = 1024
n_wires, n_layers = 4, 3
ql_dev = qml.device("default.mixed", wires=n_wires)

@qml.qnode(ql_dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_wires))
    p = float(noise_factor_base * LAMBDA_SCALE)
    for w in range(n_wires):
        qml.DepolarizingChannel(p,wires=w); qml.BitFlip(p,wires=w)
        qml.PhaseFlip(p,wires=w); qml.AmplitudeDamping(p,wires=w); qml.PhaseDamping(p,wires=w)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_wires))
    for w in range(n_wires):
        qml.DepolarizingChannel(p,wires=w); qml.BitFlip(p,wires=w)
        qml.PhaseFlip(p,wires=w); qml.AmplitudeDamping(p,wires=w); qml.PhaseDamping(p,wires=w)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

weight_shapes = {"weights": (n_layers, n_wires, 3)}
class QuantumLayer(nn.Module):
    def __init__(self): super().__init__(); self.qlayer = qnn.TorchLayer(quantum_circuit, weight_shapes)
    def forward(self,x):
        out = self.qlayer(x)
        sigma = torch.sqrt(torch.clamp(1 - out**2, min=0.0) / SHOTS)
        return torch.clamp(out + torch.randn_like(out)*sigma, -1.0, 1.0)

# ---------- Model ----------
class HybridQuantumModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.sran3 = SSRAN3(); self.lvit4e = LViT4E()
        self.feature_reducer = nn.Linear(1024, n_wires)
        self.quantum = QuantumLayer()
        self.classifier = nn.Linear(n_wires, num_classes)
    def forward(self, rgb, tif):
        fr = self.sran3(rgb); ft = self.lvit4e(tif)
        q_in = self.feature_reducer(torch.cat([fr, ft], dim=1))
        q_out = self.quantum(q_in)
        return self.classifier(q_out)

# ---------- Train/Eval ----------
def train_one_epoch(model, loader, opt, crit, device):
    model.train(); tot_loss=0; corr=0; total=0
    for rgb,tif,labels in tqdm(loader, desc="Train", leave=False):
        rgb,tif,labels = rgb.to(device), tif.to(device), labels.to(device)
        logits = model(rgb,tif); loss = crit(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item()
        corr += (logits.argmax(1)==labels).sum().item(); total += labels.size(0)
    return tot_loss/len(loader), 100*corr/max(1,total)

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval(); tot_loss=0; corr=0; total=0
    for rgb,tif,labels in tqdm(loader, desc="Eval", leave=False):
        rgb,tif,labels = rgb.to(device), tif.to(device), labels.to(device)
        logits = model(rgb,tif); loss = crit(logits, labels)
        tot_loss += loss.item()
        corr += (logits.argmax(1)==labels).sum().item(); total += labels.size(0)
    return tot_loss/len(loader), 100*corr/max(1,total)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("ckpts", exist_ok=True)
    model = HybridQuantumModel().to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50  # change to 3 for a quick test

    for ep in range(num_epochs):
        print(f"\nEpoch {ep+1}/{num_epochs}")
        tr_loss,tr_acc = train_one_epoch(model, train_loader, opt, crit, device)
        va_loss,va_acc = evaluate(model, val_loader, crit, device)
        print(f"Train Acc: {tr_acc:.2f}% | Loss: {tr_loss:.4f} || Val Acc: {va_acc:.2f}% | Val Loss: {va_loss:.4f}")
        torch.save(model.state_dict(), f"ckpts/model_ep_{ep+1:03d}.pt")
    print("✔ Saved per-epoch checkpoints in ./ckpts")
