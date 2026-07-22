"""Train a compact residual-CNN on the corpus tiles (AI vs real).

High-pass residual input (image minus blur) + small convnet — the standard
recipe for generative-image forensics. Split is GROUP-aware (by parent
image) so reported accuracy is on tiles from images never seen in training.
Saves weights to aidetect/cnn_model.pt plus metrics.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "corpus"
MODEL_PATH = REPO / "aidetect" / "cnn_model.pt"
SEED = 20260722

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(8)


def residual_tensor(img: Image.Image) -> torch.Tensor:
    rgb = np.asarray(img, dtype=np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    res = (rgb - blur) / 25.0
    return torch.from_numpy(res.transpose(2, 0, 1))


class TileSet(torch.utils.data.Dataset):
    def __init__(self, files, labels, train):
        self.files, self.labels, self.train = files, labels, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.train:
            x0 = random.randint(0, max(0, img.size[0] - 128))
            y0 = random.randint(0, max(0, img.size[1] - 128))
            img = img.crop((x0, y0, x0 + 128, y0 + 128))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            w, h = img.size
            img = img.crop(((w - 128) // 2, (h - 128) // 2,
                            (w - 128) // 2 + 128, (h - 128) // 2 + 128))
        return residual_tensor(img), self.labels[i]


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_files():
    files, labels, groups = [], [], []
    for label, sub in ((1, "ai"), (0, "real")):
        for p in sorted((CORPUS / sub).glob("*.jpg")):
            files.append(p)
            labels.append(label)
            groups.append(p.name.split("__")[0])
    return files, np.array(labels), np.array(groups)


def main() -> int:
    files, labels, groups = load_files()
    parents = sorted(set(groups))
    rng = np.random.default_rng(SEED)
    rng.shuffle(parents)
    # group-aware split: alternate parents into test until ~20% of tiles,
    # keeping both classes represented
    test_parents = set()
    test_tiles = 0
    for par in parents:
        if test_tiles / len(files) >= 0.20:
            break
        test_parents.add(par)
        test_tiles += int((groups == par).sum())
    tr_idx = [i for i, g in enumerate(groups) if g not in test_parents]
    te_idx = [i for i, g in enumerate(groups) if g in test_parents]
    print(f"{len(files)} tiles from {len(parents)} parents; "
          f"train {len(tr_idx)}, test {len(te_idx)} "
          f"({int(labels[te_idx].sum())} AI / {int((1 - labels[te_idx]).sum())} real) "
          f"from {len(test_parents)} unseen parents")

    train_ds = TileSet([files[i] for i in tr_idx], labels[tr_idx].tolist(), True)
    test_ds = TileSet([files[i] for i in te_idx], labels[te_idx].tolist(), False)
    n_pos = labels[tr_idx].sum()
    n_neg = len(tr_idx) - n_pos
    class_w = torch.tensor([len(tr_idx) / (2 * n_neg), len(tr_idx) / (2 * n_pos)],
                           dtype=torch.float32)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True,
                                           num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, num_workers=4)

    model = ResidualNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    best_acc, best_state = 0.0, None
    for epoch in range(20):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=class_w)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_dl:
                pred = model(x).argmax(1)
                correct += int((pred == y).sum())
                total += len(y)
        acc = correct / total
        print(f"epoch {epoch:02d}: held-out tile acc {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save({"state_dict": best_state, "heldout_tile_accuracy": best_acc,
                "test_parents": sorted(test_parents)}, MODEL_PATH)
    print(f"\nbest held-out tile accuracy: {best_acc:.3f}")
    print(f"wrote {MODEL_PATH}")
    json.dump({"heldout_tile_accuracy": best_acc},
              open(REPO / "aidetect" / "cnn_metrics.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
