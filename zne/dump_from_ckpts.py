import os, re, argparse
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import rasterio
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import pennylane as qml
import pennylane.qnn as qnn
import numpy as np

ap = argparse.ArgumentParser(description="DUMP ONLY: load each epoch ckpt, dump deterministic q for one λ.")
ap.add_argument("--ckpt_dir", default="ckpts")
ap.add_argument("--lambda_scale", type=float, required=True)  # e.g., 0,1,3,5
ap.add_argument("--split", choices=["train","val","test"], default="val")
ap.add_argument("--out_dir", default="zne_dumps")
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

print("\n=== DUMPER START ===")
print(f"λ={args.lambda_scale} | split={args.split} | ckpt_dir={args.ckpt_dir} | out_dir={args.out_dir} | batch_size={args.batch_size}")

# ---------- Data ----------
class EuroSATMultimodalDataset(Dataset):
    def __init__(self, csv_file, rgb_root, tif_root, transform_rgb=None, transform_tif=None):
        self.data = pd.read_csv(csv_file)
        self.rgb_root, self.tif_root = rgb_root, tif_root
        self.transform_rgb, self.transform_tif = transform_rgb, transform_tif
        self.filenames = self.data["Filename"].tolist()
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        rgb_path = os.path.join(self.rgb_root, row['Filename'])
        tif_path = os.path.join(self.tif_root, row['Filename'].replace('.jpg', '.tif'))
        rgb = Image.open(rgb_path).convert("RGB")
        if self.transform_rgb: rgb = self.transform_rgb(rgb)
        with rasterio.open(tif_path) as tif: arr = tif.read()
        arr = torch.tensor(arr/10000.0, dtype=torch.float32)
        if self.transform_tif: arr = self.transform_tif(arr)
        return rgb, arr, int(row['Label'])

transform_rgb = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
class ResizeTIF:
    def __init__(self, size=(224,224)): self.size=size
    def __call__(self,x): 
        return torch.nn.functional.interpolate(x.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
transform_tif = ResizeTIF((224,224))

base = os.path.expanduser("~/quantum/jsac/EuroSAT_extracted")
rgb_root = base
tif_root = os.path.expanduser("~/quantum/jsac/EuroSAT_extracted/allBands")
csv_train = os.path.join(base,"train.csv")
csv_val   = os.path.join(base,"validation.csv")
csv_test  = os.path.join(base,"test.csv")
split_map = {"train": csv_train, "val": csv_val, "test": csv_test}
csv_used = split_map[args.split]

ds = EuroSATMultimodalDataset(csv_used, rgb_root, tif_root, transform_rgb, transform_tif)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
print(f"Dataset: {args.split} | samples={len(ds)}")

# ---------- Model ----------
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

noise_factor_base = 0.02        # ↓↓↓ match training
LAMBDA_SCALE = float(args.lambda_scale)
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
        super().__init__(); 
        self.qlayer = qnn.TorchLayer(quantum_circuit, weight_shapes)
    @torch.no_grad()
    def forward_no_jitter(self,x): 
        return self.qlayer(x)

class HybridQuantumModel(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.sran3=SSRAN3(); self.lvit4e=LViT4E()
        self.feature_reducer=nn.Linear(1024,n_wires)
        self.quantum=QuantumLayer()
        self.classifier=nn.Linear(n_wires,num_classes)
    @torch.no_grad()
    def eval_qouts(self, rgb, tif):
        fr=self.sran3(rgb); ft=self.lvit4e(tif)
        q_in=self.feature_reducer(torch.cat([fr,ft],dim=1))
        q_out=self.quantum.forward_no_jitter(q_in)  # deterministic
        logits=self.classifier(q_out)
        return q_out, logits

def list_ckpts(ckpt_dir):
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    return [os.path.join(ckpt_dir, f) for f in files]

def dump_for_ckpt(ckpt_path, out_dir, split_name, device):
    m = re.search(r'ep_(\d+)', os.path.basename(ckpt_path))
    ep = int(m.group(1)) if m else -1
    model = HybridQuantumModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rows=[]; idx_ptr=0
    for rgb, tif, labels in tqdm(loader, desc=f"Dump ep{ep:03d} λ={LAMBDA_SCALE}", leave=False):
        B=labels.size(0)
        names = ds.filenames[idx_ptr:idx_ptr+B]; idx_ptr += B
        rgb,tif = rgb.to(device), tif.to(device)
        q_out,_ = model.eval_qouts(rgb,tif)
        q = q_out.cpu().numpy()
        for i in range(B):
            rows.append({"epoch":ep,"filename":names[i],"label":int(labels[i]),
                         "q0":float(q[i,0]),"q1":float(q[i,1]),"q2":float(q[i,2]),"q3":float(q[i,3])})

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"dump_{split_name}_ep{ep:03d}_lam{int(LAMBDA_SCALE)}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # save head weights for this epoch
    W = model.classifier.weight.detach().cpu().numpy()
    b = model.classifier.bias.detach().cpu().numpy()
    np.savez(os.path.join(out_dir, f"head_weights_ep{ep:03d}.npz"), W=W, b=b)
    print(f"✔ ep {ep}: wrote {csv_path} and head_weights_ep{ep:03d}.npz")

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = list_ckpts(args.ckpt_dir)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.ckpt_dir}.")
    print(f"Found {len(ckpts)} checkpoints. First: {os.path.basename(ckpts[0])}")
    if args.check:
        print("CHECK ONLY: not dumping. Exiting."); raise SystemExit(0)
    for ck in ckpts:
        dump_for_ckpt(ck, args.out_dir, args.split, device)
    print("=== DUMPER DONE ===\n")
