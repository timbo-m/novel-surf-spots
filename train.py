"""Fine-tune a pretrained CNN on your labelled chips.

  python train.py --chips chips --epochs 15 --out model.pt

Several chip folders (e.g. different days) can be passed: --chips day1 day2
Each needs a labels.csv from label.py.
"""
import argparse, csv, os, random
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import CLASSES, build_model, device, eval_transform, train_transform


class Chips(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


def load_items(dirs):
    items = []
    for d in dirs:
        with open(os.path.join(d, "labels.csv")) as f:
            for r in csv.DictReader(f):
                if r["label"] in CLASSES:
                    items.append((os.path.join(d, r["chip"]), CLASSES.index(r["label"])))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", nargs="+", default=["chips"])
    ap.add_argument("--arch", default="convnext_tiny", choices=["convnext_tiny", "resnet50", "resnet18"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=float, default=0.2, help="fraction held out for validation")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline testing only)")
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    items = load_items(args.chips)
    random.seed(0); random.shuffle(items)
    n_val = max(1, int(len(items) * args.val))
    val, train = items[:n_val], items[n_val:]
    counts = Counter(y for _, y in train)
    print("train:", {CLASSES[k]: v for k, v in sorted(counts.items())}, " val:", len(val))

    dev = device()
    model = build_model(args.arch, pretrained=not args.no_pretrained).to(dev)
    # Clean waves will be rare compared to flat sea, so weight classes by inverse frequency.
    w = torch.tensor([len(train) / (len(CLASSES) * max(counts[c], 1)) for c in range(len(CLASSES))])
    loss_fn = torch.nn.CrossEntropyLoss(weight=w.to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    tl = DataLoader(Chips(train, train_transform()), args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(Chips(val, eval_transform()), args.batch, num_workers=2)
    best = -1
    for ep in range(args.epochs):
        model.train(); tot = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(y)
        sched.step()

        model.eval(); conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
        with torch.no_grad():
            for x, y in vl:
                p = model(x.to(dev)).argmax(1).cpu()
                for a, b in zip(y, p):
                    conf[a, b] += 1
        acc = conf.diag().sum().item() / max(conf.sum().item(), 1)
        clean_recall = conf[0, 0].item() / max(conf[0].sum().item(), 1)
        print(f"epoch {ep+1:2d}  loss {tot/len(train):.3f}  val acc {acc:.2f}  clean recall {clean_recall:.2f}")
        score = acc + clean_recall
        if score > best:
            best = score
            torch.save({"arch": args.arch, "classes": CLASSES, "state": model.state_dict()}, args.out)
    print("confusion (rows=true, cols=pred):", CLASSES); print(conf)
    print("saved best model ->", args.out)


if __name__ == "__main__":
    main()
