# file: dump_from_ckpts_cifar.py
import os, re, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T
from tqdm import tqdm
import pandas as pd
import pennylane as qml
import pennylane.qnn as qnn

ap = argparse.ArgumentParser(description="CIFAR10 DUMPER: load each epoch ckpt, dump deterministic q for one λ.")
ap.add_argument("--ckpt_dir", default="ckpts_cifar")
ap.add_argument("--lambda_scale", type=float, required=True)  # e.g., 0,1,3,5
ap.add_argument("--out_dir", default="zne_dumps_cifar")
ap.add_argument("--batch_size", type=int, default=256)
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

print("\n=== CIFAR DUMPER START ===")
print(f"λ={args.lambda_scale} | ckpt_dir={args.ckpt_dir} | out_dir={args.out_dir} | batch_size={args.batch_size}")

# ---------- CIFAR-10 test data ----------
IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD  = [0.229, 0.224, 0.225]
transform_eval = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])
root = os.path.expanduser("~/data/cifar10")
test_set = datasets.CIFAR10(root=root, train=False, transform=transform_eval, download=True)

# Stable “filename” key for merging: id_{index}
filenames = [f"id_{i:05d}" for i in range(len(test_set))]
labels_np = np.array([test_set[i][1] for i in range(len(test_set))], dtype=int)

loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
print(f"Dataset: CIFAR-10 test | samples={len(test_set)}")

# ---------- Model (same as training) ----------
class ResBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x): return self.backbone(x).view(x.size(0), -1)

noise_factor_base = 0.02   # match training
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
        super().__init__(); self.qlayer = qnn.TorchLayer(quantum_circuit, weight_shapes)
    @torch.no_grad()
    def forward_no_jitter(self, x): return self.qlayer(x)

class HybridQuantumModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = ResBackbone()
        self.feature_reducer = nn.Linear(512, n_wires)
        self.quantum = QuantumLayer()
        self.classifier = nn.Linear(n_wires, num_classes)
    @torch.no_grad()
    def eval_qouts(self, x):
        f = self.backbone(x)
        q_in = self.feature_reducer(f)
        q_out = self.quantum.forward_no_jitter(q_in)
        logits = self.classifier(q_out)
        return q_out, logits

def list_ckpts(ckpt_dir):
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    return [os.path.join(ckpt_dir, f) for f in files]

def dump_for_ckpt(ckpt_path, out_dir, device):
    m = re.search(r'ep_(\d+)', os.path.basename(ckpt_path))
    ep = int(m.group(1)) if m else -1
    model = HybridQuantumModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rows=[]; idx_ptr=0
    for imgs, labels in tqdm(loader, desc=f"Dump ep{ep:03d} λ={LAMBDA_SCALE}", leave=False):
        B = labels.size(0)
        names = filenames[idx_ptr:idx_ptr+B]; idx_ptr += B
        imgs = imgs.to(device)
        q_out,_ = model.eval_qouts(imgs)
        q = q_out.cpu().numpy()
        for i in range(B):
            rows.append({"epoch":ep,"filename":names[i],"label":int(labels[i]),
                         "q0":float(q[i,0]),"q1":float(q[i,1]),"q2":float(q[i,2]),"q3":float(q[i,3])})

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"dump_test_ep{ep:03d}_lam{int(LAMBDA_SCALE)}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

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
        dump_for_ckpt(ck, args.out_dir, device)
    print("=== DUMPER DONE ===\n")
