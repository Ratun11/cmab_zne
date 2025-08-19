# file: train_save_epochs_cifar.py
import os, random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
from tqdm import tqdm

import pennylane as qml
from pennylane import numpy as np
import pennylane.qnn as qnn

# ----- Repro -----
SEED = 123
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything(SEED)
torch.set_default_dtype(torch.float32)

# ---------- CIFAR-10 data ----------
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]

transform_train = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])

transform_eval = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])

root = os.path.expanduser("~/data/cifar10")
train_full = datasets.CIFAR10(root=root, train=True,  transform=transform_train, download=True)
test_set   = datasets.CIFAR10(root=root, train=False, transform=transform_eval,  download=True)

# Make a small validation split out of train (e.g., 45k/5k)
val_size = 5000
train_size = len(train_full) - val_size
train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# ---------- Model: ResNet18 backbone + quantum head ----------
class ResBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> [B,512,1,1]
    def forward(self, x): return self.backbone(x).view(x.size(0), -1)  # [B,512]

# Quantum settings
noise_factor_base = 0.05     # mild base noise so λ=3/5 are meaningful
LAMBDA_SCALE = 1.0           # train at λ=1
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
    def __init__(self):
        super().__init__()
        self.qlayer = qnn.TorchLayer(quantum_circuit, weight_shapes)
    def forward(self, x):
        out = self.qlayer(x)  # [B,4]
        # pseudo-shot jitter (no-op feel)
        sigma = torch.sqrt(torch.clamp(1 - out**2, min=0.0) / SHOTS)
        return torch.clamp(out + torch.randn_like(out)*sigma, -1.0, 1.0)

class HybridQuantumModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = ResBackbone()            # [B,512]
        self.feature_reducer = nn.Linear(512, n_wires)
        self.quantum = QuantumLayer()
        self.classifier = nn.Linear(n_wires, num_classes)
    def forward(self, x):
        f = self.backbone(x)                     # [B,512]
        q_in = self.feature_reducer(f)           # [B,4]
        q_out = self.quantum(q_in)               # [B,4]
        return self.classifier(q_out)            # [B,10]

# ---------- Train/Eval ----------
def train_one_epoch(model, loader, opt, crit, device):
    model.train(); tot_loss=0; corr=0; total=0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs); loss = crit(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item()
        corr += (logits.argmax(1)==labels).sum().item(); total += labels.size(0)
    return tot_loss/len(loader), 100*corr/max(1,total)

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval(); tot_loss=0; corr=0; total=0
    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs); loss = crit(logits, labels)
        tot_loss += loss.item()
        corr += (logits.argmax(1)==labels).sum().item(); total += labels.size(0)
    return tot_loss/len(loader), 100*corr/max(1,total)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("ckpts_cifar", exist_ok=True)
    model = HybridQuantumModel(num_classes=10).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50   # set to 3 for a quick test
    for ep in range(num_epochs):
        print(f"\nEpoch {ep+1}/{num_epochs}")
        tr_loss,tr_acc = train_one_epoch(model, train_loader, opt, crit, device)
        va_loss,va_acc = evaluate(model, val_loader,   crit, device)
        print(f"Train Acc: {tr_acc:.2f}% | Loss: {tr_loss:.4f} || Val Acc: {va_acc:.2f}% | Val Loss: {va_loss:.4f}")
        torch.save(model.state_dict(), f"ckpts_cifar/model_ep_{ep+1:03d}.pt")
    print("✔ Saved per-epoch checkpoints in ./ckpts_cifar")
