"""Training loop + evaluation. bf16 autocast, grad accumulation, resume, CSV logging.

Protocol: FIXED epoch budget (cfg.epochs) for every model/dataset -> comparable runs.
Reported numbers = FINAL-epoch metrics. Best-on-test is tracked as a diagnostic only
(never report it: checkpoint selection by test metrics is test peeking).
Early stopping exists but is off unless cfg.early_stop_patience > 0.
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch


def confusion(pred, target, num_classes):
    m = target >= 0
    p, t = pred[m], target[m]
    return np.bincount(t * num_classes + p, minlength=num_classes ** 2).reshape(num_classes,
                                                                                num_classes)

def metrics_from_confusion(cm):
    total = cm.sum()
    oa = np.diag(cm).sum() / max(total, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(1)
        iou = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))
    present = cm.sum(1) > 0
    aa = np.nanmean(per_class_acc[present])
    miou = np.nanmean(iou[present])
    pe = (cm.sum(0) * cm.sum(1)).sum() / max(total ** 2, 1)
    kappa = (oa - pe) / (1 - pe + 1e-12)
    return {"oa": float(oa), "aa": float(aa), "kappa": float(kappa), "miou": float(miou),
            "per_class_iou": [float(x) for x in iou]}


@torch.no_grad()
def evaluate_dense(model, cube, labels, test_idx, num_classes, device, amp_dtype):
    model.eval()
    x = cube.unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
        logits = model(x)
    pred = logits.argmax(1).squeeze(0).cpu().numpy().ravel()
    target = np.full_like(pred, -1)
    target[test_idx] = labels.ravel()[test_idx]
    return metrics_from_confusion(confusion(pred, target, num_classes))


@torch.no_grad()
def evaluate_dense_tiled(model, cube, labels, test_idx, num_classes, device, amp_dtype,
                         tile=64, stride=32):
    """Sliding-window inference for tile-trained models (matches training distribution)."""
    model.eval()
    x = cube.unsqueeze(0).to(device)
    _, _, H, W = x.shape
    th, tw = min(tile, H), min(tile, W)
    logits = torch.zeros(1, num_classes, H, W, device=device)
    counts = torch.zeros(1, 1, H, W, device=device)
    ys = sorted(set(list(range(0, max(H - th, 0) + 1, stride)) + [max(H - th, 0)]))
    xs = sorted(set(list(range(0, max(W - tw, 0) + 1, stride)) + [max(W - tw, 0)]))
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
        for yy in ys:
            for xx in xs:
                out = model(x[:, :, yy:yy + th, xx:xx + tw]).float()
                logits[:, :, yy:yy + th, xx:xx + tw] += out
                counts[:, :, yy:yy + th, xx:xx + tw] += 1
    pred = (logits / counts).argmax(1).squeeze(0).cpu().numpy().ravel()
    target = np.full_like(pred, -1)
    target[test_idx] = labels.ravel()[test_idx]
    return metrics_from_confusion(confusion(pred, target, num_classes))


def train_dense(model, loader, scene, test_idx, criterion, cfg, run_dir: Path, device="cuda"):
    """Returns {"final": metrics_at_last_epoch, "best": best_on_test_diagnostic}."""
    run_dir.mkdir(parents=True, exist_ok=True)
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    enc_mult = float(getattr(cfg, "enc_lr_mult", 1.0))
    if enc_mult != 1.0:
        enc_p, head_p = [], []
        for _n, _pp in model.named_parameters():
            (head_p if _n.startswith("head") else enc_p).append(_pp)
        opt = torch.optim.AdamW(
            [{"params": enc_p, "lr": cfg.lr * enc_mult},
             {"params": head_p, "lr": cfg.lr}],
            weight_decay=cfg.weight_decay)
        print(f"discriminative LR: encoder {cfg.lr*enc_mult:.2e}, head {cfg.lr:.2e}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    model.to(device)
    criterion.to(device)

    start_epoch, best, bad_evals = 0, {"miou": -1.0}, 0
    eval_hist = []
    ckpt_path = run_dir / "last.pt"
    if cfg.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"]); start_epoch = state["epoch"] + 1
        best = state["best"]
        print(f"resumed from epoch {start_epoch}")

    cube_t = torch.from_numpy(scene.cube).permute(2, 0, 1).contiguous()
    tiled = getattr(cfg, "mode", "whole_scene") == "tiles"

    def _eval():
        if tiled:
            return evaluate_dense_tiled(model, cube_t, scene.labels, test_idx,
                                        scene.num_classes, device, amp_dtype,
                                        tile=cfg.tile_size, stride=cfg.tile_size // 2)
        return evaluate_dense(model, cube_t, scene.labels, test_idx,
                              scene.num_classes, device, amp_dtype)
    log_path = run_dir / "log.csv"
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if x.dim() == 3:
                x, y = x.unsqueeze(0), y.unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
                loss = criterion(model(x), y) / cfg.grad_accum
            loss.backward()
            if (i + 1) % cfg.grad_accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.item() * cfg.grad_accum)
        sched.step()

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            m = _eval()
            eval_hist.append(m)
            if m["miou"] > best["miou"]:
                best = {**m, "epoch": epoch}
                torch.save(model.state_dict(), run_dir / "best_diag.pt")
                bad_evals = 0
            else:
                bad_evals += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "best": best}, ckpt_path)
            row = {"epoch": epoch, "loss": float(np.mean(losses)), "oa": m["oa"],
                   "aa": m["aa"], "kappa": m["kappa"], "miou": m["miou"],
                   "sec_per_epoch": time.time() - t0}
            write_csv_row(log_path, row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            if cfg.early_stop_patience > 0 and bad_evals * cfg.eval_every >= cfg.early_stop_patience:
                print("early stop"); break

    final = _eval()
    final["epoch"] = epoch
    eval_hist.append(final)
    _last = eval_hist[-5:]
    final_avg = {"n_avg": len(_last)}
    for _k, _v in final.items():
        if _k == "epoch" or not all(_k in h for h in _last):
            continue
        if _k == "per_class_iou":
            final_avg[_k] = [float(sum(c) / len(_last))
                             for c in zip(*[h[_k] for h in _last])]
        elif isinstance(_v, (int, float)):
            final_avg[_k] = float(sum(h[_k] for h in _last) / len(_last))
    torch.save(model.state_dict(), run_dir / "final.pt")
    print("FINAL(report):", {k: round(v, 4) for k, v in final.items() if k != "per_class_iou"},
          "| best-on-test(diagnostic): miou", round(best["miou"], 4))
    print("FINAL(avg of last %d evals): miou" % final_avg["n_avg"],
          round(final_avg["miou"], 4))
    return {"final": final, "final_avg": final_avg, "best": best}


def write_csv_row(path: Path, row: dict):
    """Schema-safe append: if columns changed, rewrite the file with the union."""
    path = Path(path)
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                r.pop(None, None)
                rows.append(r)
    rows.append({k: str(v) for k, v in row.items()})
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(rows)


def profile_model(model, in_bands, hw, device="cuda", amp_dtype=torch.bfloat16):
    model.to(device).eval()
    x = torch.randn(1, in_bands, *hw, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        for _ in range(3):
            model(x)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
    return {"params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "peak_vram_GB": torch.cuda.max_memory_allocated() / 1e9,
            "latency_ms": (time.time() - t0) / 10 * 1000}
