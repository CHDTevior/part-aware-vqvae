#!/usr/bin/env python3
import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import dataset_TM_eval, dataset_VQ
from models.evaluator_wrapper import EvaluatorModelWrapper
from models.vqvae import HumanVQVAE
from options.get_eval_option import get_opt
from utils.losses import ReConsLoss
from utils.word_vectorizer import WordVectorizer
import utils.eval_trans as eval_trans


class CappedBuffer:
    def __init__(self, cap: int):
        self.cap = cap
        self.data = None

    def add(self, x: torch.Tensor) -> None:
        x = x.detach().cpu()
        if x.numel() == 0:
            return
        if self.data is None:
            self.data = x[: self.cap]
            return
        merged = torch.cat([self.data, x], dim=0)
        if merged.shape[0] <= self.cap:
            self.data = merged
        else:
            idx = torch.randperm(merged.shape[0])[: self.cap]
            self.data = merged[idx]

    def get(self) -> torch.Tensor:
        if self.data is None:
            return torch.empty(0)
        return self.data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline VQ-VAE evaluation + latent visualization")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pth) with key 'net' or raw state_dict")
    parser.add_argument("--out-dir", type=str, default="offline_eval", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--dataname", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--partition-file", type=str, default="./partition_analysis/skeleton_partition.json")
    parser.add_argument("--quantizer", type=str, default="ema_reset", choices=["ema_reset", "ema", "orig", "reset"])
    parser.add_argument("--nb-code", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=128)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--output-emb-width", type=int, default=128)
    parser.add_argument("--vq-act", type=str, default="relu")
    parser.add_argument("--vq-norm", type=str, default=None)
    parser.add_argument("--strict-load", action="store_true", help="Use strict=True when loading checkpoint")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=40, help="Batches for offline diagnostics")
    parser.add_argument("--max-points", type=int, default=8000, help="Max points stored for each visualization set")
    parser.add_argument("--nn-samples", type=int, default=128, help="Samples for nearest-neighbor collapse check")
    parser.add_argument("--run-training-style-eval", action="store_true", help="Run eval_trans.evaluation_vqvae (FID/diversity/R-precision)")
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("offline_vq_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_state(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "net" in ckpt and isinstance(ckpt["net"], dict):
        state = ckpt["net"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Checkpoint format not supported")
    return strip_module_prefix(state)


def build_model(args: argparse.Namespace, device: torch.device) -> HumanVQVAE:
    if device.type == "cpu" and args.quantizer in {"ema_reset", "ema"}:
        raise RuntimeError("quantizer ema/ema_reset uses CUDA-only codebook init in this codebase; run with --device cuda")

    model_args = SimpleNamespace(
        dataname=args.dataname,
        quantizer=args.quantizer,
        partition_file=args.partition_file if args.partition_file else None,
        mu=0.99,
        beta=1.0,
    )
    model = HumanVQVAE(
        model_args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
        args.vq_act,
        args.vq_norm,
    )
    model.to(device)
    model.eval()
    return model


def pca2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        return torch.empty(0, 2)
    if x.shape[0] == 0:
        return torch.empty(0, 2)
    if x.shape[0] < 2:
        return torch.cat([x[:, :1], torch.zeros((x.shape[0], 1), dtype=x.dtype)], dim=1)
    x = x - x.mean(dim=0, keepdim=True)
    q = min(2, x.shape[1], x.shape[0] - 1)
    if q <= 0:
        return torch.zeros((x.shape[0], 2))
    _, _, v = torch.pca_lowrank(x, q=q)
    proj = x @ v[:, :q]
    if q == 1:
        proj = torch.cat([proj, torch.zeros_like(proj)], dim=1)
    return proj


def scatter_by_part(points_2d: np.ndarray, part_ids: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    n_parts = int(part_ids.max()) + 1 if part_ids.size > 0 else 1
    cmap = plt.get_cmap("tab10", n_parts)
    for p in range(n_parts):
        mask = part_ids == p
        if not np.any(mask):
            continue
        plt.scatter(points_2d[mask, 0], points_2d[mask, 1], s=4, alpha=0.45, color=cmap(p), label=f"part{p}")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=3, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_usage_heatmap(usage: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(12, 3.2))
    plt.imshow(np.log1p(usage), aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="log(1+count)")
    plt.xlabel("Code Index")
    plt.ylabel("Part")
    plt.title("Code Usage Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_nn_hist(nn_ratio: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(nn_ratio, bins=30)
    plt.axvline(1.0, linestyle="--", color="red", linewidth=1.2, label="ratio=1.0")
    plt.xlabel("MSE(nn_recon, x) / MSE(self_recon, x)")
    plt.ylabel("Count")
    plt.title("Nearest Neighbor Reconstruction Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_loss_scatter(recon_vals: np.ndarray, commit_vals: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(recon_vals, commit_vals, s=8, alpha=0.6)
    plt.xlabel("Recon Loss")
    plt.ylabel("Commit Loss")
    plt.title("Recon vs Commit Loss (per batch)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def get_codebook_tensor(q: torch.nn.Module) -> torch.Tensor:
    if hasattr(q, "codebook"):
        cb = q.codebook
        if isinstance(cb, torch.nn.Parameter):
            cb = cb.data
        return cb.detach()
    if hasattr(q, "embedding"):
        return q.embedding.weight.detach()
    raise RuntimeError(f"Unknown quantizer type: {q.__class__.__name__}")


def get_num_codes(q: torch.nn.Module) -> int:
    if hasattr(q, "nb_code"):
        return int(q.nb_code)
    if hasattr(q, "n_e"):
        return int(q.n_e)
    raise RuntimeError(f"Cannot determine codebook size for quantizer {q.__class__.__name__}")


@torch.no_grad()
def extract_latent_stats(
    model: HumanVQVAE,
    x: torch.Tensor,
    encoder_bufs: List[CappedBuffer],
    quant_bufs: List[CappedBuffer],
    usage_counts: List[torch.Tensor],
    descriptor_list: List[torch.Tensor],
    motion_list: List[torch.Tensor],
    recon_list: List[torch.Tensor],
    nn_cap: int,
) -> None:
    vq = model.vqvae
    x_in = vq.preprocess(x)
    x_in_t = x_in.permute(0, 2, 1).contiguous()

    desc_parts = []
    bsz = x.shape[0]

    for p, part in enumerate(vq.partSeg):
        x_part = x_in_t[:, :, part].permute(0, 2, 1).contiguous()
        z_e = vq.limb_encoders[p](x_part)  # B, D, T'
        z_e_t = z_e.permute(0, 2, 1).contiguous()  # B, T', D
        z_flat = z_e_t.view(-1, z_e_t.shape[-1])

        idx_flat = vq.quantizers[p].quantize(z_flat)
        usage_counts[p] += torch.bincount(idx_flat.detach().cpu(), minlength=get_num_codes(vq.quantizers[p]))

        z_q = vq.quantizers[p].dequantize(idx_flat).view(z_e_t.shape)  # B, T', D
        z_q_flat = z_q.reshape(-1, z_q.shape[-1])

        encoder_bufs[p].add(z_flat)
        quant_bufs[p].add(z_q_flat)

        desc_parts.append(z_e_t.mean(dim=1))  # B, D

    if len(descriptor_list) < nn_cap:
        desc = torch.cat(desc_parts, dim=1).detach().cpu()  # B, n_parts*D
        x_out, _, _ = model(x, type="full")
        need = nn_cap - len(descriptor_list)
        take = min(need, bsz)
        perm = torch.randperm(bsz)[:take]
        for i in perm.tolist():
            descriptor_list.append(desc[i])
            motion_list.append(x[i].detach().cpu())
            recon_list.append(x_out[i].detach().cpu())


def calc_usage_metrics(usage: torch.Tensor) -> Dict[str, float]:
    total = usage.sum().item()
    if total <= 0:
        return {"perplexity": 0.0, "dead_codes": float(usage.shape[0]), "active_ratio": 0.0}
    prob = usage.float() / total
    perplexity = torch.exp(-(prob[prob > 0] * torch.log(prob[prob > 0])).sum()).item()
    dead = (usage == 0).sum().item()
    active_ratio = 1.0 - dead / usage.shape[0]
    return {"perplexity": perplexity, "dead_codes": float(dead), "active_ratio": active_ratio}


def maybe_run_training_style_eval(args: argparse.Namespace, model: HumanVQVAE, logger: logging.Logger) -> Dict[str, float]:
    if not args.run_training_style_eval:
        return {}

    logger.info("Running training-style evaluation (FID/diversity/R-precision)...")
    if args.dataname == "kit":
        dataset_opt_path = "checkpoints/kit/Comp_v6_KLD005/opt.txt"
    else:
        dataset_opt_path = "checkpoints/t2m/Comp_v6_KLD005/opt.txt"

    w_vectorizer = WordVectorizer("./glove", "our_vab")
    val_loader = dataset_TM_eval.DATALoader(
        args.dataname,
        False,
        32,
        w_vectorizer,
        unit_length=2 ** args.down_t,
        num_workers=args.num_workers,
        shuffle=True,
    )
    wrapper_opt = get_opt(dataset_opt_path, torch.device(args.device))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    best_fid, _, best_div, best_top1, best_top2, best_top3, best_matching, _, _ = eval_trans.evaluation_vqvae(
        out_dir=args.out_dir,
        val_loader=val_loader,
        net=model,
        logger=logger,
        writer=None,
        nb_iter=0,
        best_fid=1000,
        best_iter=0,
        best_div=100,
        best_top1=0,
        best_top2=0,
        best_top3=0,
        best_matching=100,
        eval_wrapper=eval_wrapper,
        draw=False,
        save=False,
    )

    return {
        "fid": float(best_fid),
        "diversity": float(best_div),
        "top1": float(best_top1),
        "top2": float(best_top2),
        "top3": float(best_top3),
        "matching_score": float(best_matching),
    }


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.out_dir = str(Path(args.out_dir).resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    logger.info("Loading model and checkpoint...")
    model = build_model(args, device)
    state = load_state(args.ckpt)
    missing, unexpected = model.load_state_dict(state, strict=args.strict_load)
    if args.strict_load and (missing or unexpected):
        raise RuntimeError("strict load enabled but checkpoint mismatch found")
    if not args.strict_load:
        logger.info("Non-strict load. missing=%d unexpected=%d", len(missing), len(unexpected))
    model.eval()

    logger.info("Preparing dataloader...")
    train_loader = dataset_VQ.DATALoader(
        args.dataname,
        args.batch_size,
        num_workers=args.num_workers,
        window_size=args.window_size,
        unit_length=2 ** args.down_t,
    )

    n_parts = model.vqvae.num_parts
    usage_counts = [torch.zeros(get_num_codes(model.vqvae.quantizers[i]), dtype=torch.long) for i in range(n_parts)]
    encoder_bufs = [CappedBuffer(max(1, args.max_points // n_parts)) for _ in range(n_parts)]
    quant_bufs = [CappedBuffer(max(1, args.max_points // n_parts)) for _ in range(n_parts)]
    descriptors: List[torch.Tensor] = []
    motions: List[torch.Tensor] = []
    recons: List[torch.Tensor] = []

    loss_fn = ReConsLoss("l1_smooth", 22 if args.dataname == "t2m" else 21)
    recon_vals = []
    commit_vals = []
    vel_vals = []
    perplex_vals = []
    sample_codes = []

    logger.info("Collecting offline diagnostics for %d batches...", args.num_batches)
    with torch.no_grad():
        for i, motion in enumerate(train_loader):
            if i >= args.num_batches:
                break
            x = motion.to(device).float()

            x_out, loss_commit, perplexity = model(x, type="full")
            recon = loss_fn(x_out, x).item()
            vel = loss_fn.forward_joint(x_out, x).item()
            recon_vals.append(recon)
            commit_vals.append(float(loss_commit.item()))
            vel_vals.append(vel)
            perplex_vals.append(float(perplexity.item()))

            codes = model(x, type="encode")
            sample_codes.append(codes.detach().cpu())

            extract_latent_stats(
                model,
                x,
                encoder_bufs,
                quant_bufs,
                usage_counts,
                descriptors,
                motions,
                recons,
                args.nn_samples,
            )

    if not recon_vals:
        raise RuntimeError("No batch processed. Check dataset path/symlink.")

    encoder_points = []
    encoder_part = []
    quant_points = []
    quant_part = []
    for p in range(n_parts):
        ep = encoder_bufs[p].get()
        qp = quant_bufs[p].get()
        if ep.numel() > 0:
            encoder_points.append(ep)
            encoder_part.append(torch.full((ep.shape[0],), p, dtype=torch.long))
        if qp.numel() > 0:
            quant_points.append(qp)
            quant_part.append(torch.full((qp.shape[0],), p, dtype=torch.long))
    encoder_points = torch.cat(encoder_points, dim=0) if encoder_points else torch.empty(0, args.code_dim)
    quant_points = torch.cat(quant_points, dim=0) if quant_points else torch.empty(0, args.code_dim)
    encoder_part = torch.cat(encoder_part, dim=0) if encoder_part else torch.empty(0, dtype=torch.long)
    quant_part = torch.cat(quant_part, dim=0) if quant_part else torch.empty(0, dtype=torch.long)

    logger.info("Rendering visualization images...")
    enc_2d = pca2(encoder_points.float()).numpy()
    q_2d = pca2(quant_points.float()).numpy()
    scatter_by_part(enc_2d, encoder_part.numpy(), "Encoder Latent (Pre-Quant) PCA", out_dir / "encoder_latent_pca.png")
    scatter_by_part(q_2d, quant_part.numpy(), "Quantized Latent PCA", out_dir / "quantized_latent_pca.png")

    usage_np = torch.stack(usage_counts, dim=0).numpy()
    save_usage_heatmap(usage_np, out_dir / "code_usage_heatmap.png")
    save_loss_scatter(np.array(recon_vals), np.array(commit_vals), out_dir / "recon_vs_commit.png")

    # Codebook visualization
    codebook_vecs = []
    codebook_part = []
    codebook_used = []
    for p in range(n_parts):
        cb = get_codebook_tensor(model.vqvae.quantizers[p]).detach().cpu().float()
        used_mask = usage_counts[p] > 0
        codebook_vecs.append(cb)
        codebook_part.append(torch.full((cb.shape[0],), p, dtype=torch.long))
        codebook_used.append(used_mask.long())
    codebook_vecs = torch.cat(codebook_vecs, dim=0)
    codebook_part = torch.cat(codebook_part, dim=0)
    codebook_used = torch.cat(codebook_used, dim=0).numpy()
    cb_2d = pca2(codebook_vecs).numpy()

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10", n_parts)
    for p in range(n_parts):
        mask_p = codebook_part.numpy() == p
        used_p = np.logical_and(mask_p, codebook_used == 1)
        dead_p = np.logical_and(mask_p, codebook_used == 0)
        if np.any(used_p):
            plt.scatter(cb_2d[used_p, 0], cb_2d[used_p, 1], s=14, alpha=0.75, color=cmap(p), label=f"part{p}-used")
        if np.any(dead_p):
            plt.scatter(cb_2d[dead_p, 0], cb_2d[dead_p, 1], s=18, alpha=0.9, color=cmap(p), marker="x", label=f"part{p}-dead")
    plt.title("Codebook Embeddings PCA (x = dead code)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=1.3, fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "codebook_pca.png", dpi=180)
    plt.close()

    # NN collapse check
    nn_metrics = {}
    if len(descriptors) >= 2:
        d = torch.stack(descriptors, dim=0).float()
        x = torch.stack(motions, dim=0).float()
        x_hat = torch.stack(recons, dim=0).float()
        dist = torch.cdist(d, d)
        dist.fill_diagonal_(float("inf"))
        nn_idx = dist.argmin(dim=1)
        self_mse = ((x_hat - x) ** 2).mean(dim=(1, 2))
        nn_mse = ((x_hat[nn_idx] - x) ** 2).mean(dim=(1, 2))
        ratio = (nn_mse / (self_mse + 1e-8)).numpy()
        save_nn_hist(ratio, out_dir / "nn_recon_ratio_hist.png")
        nn_metrics = {
            "nn_ratio_mean": float(np.mean(ratio)),
            "nn_ratio_median": float(np.median(ratio)),
            "nn_ratio_p90": float(np.percentile(ratio, 90)),
            "descriptor_nn_dist_mean": float(dist.min(dim=1).values.mean().item()),
            "unique_nn_fraction": float(torch.unique(nn_idx).numel() / nn_idx.numel()),
        }

    # Sequence-level code collapse (duplicates)
    codes_all = torch.cat(sample_codes, dim=0)
    uniq = torch.unique(codes_all, dim=0).shape[0]
    code_dup_ratio = 1.0 - (uniq / max(codes_all.shape[0], 1))

    usage_metrics = {}
    for p in range(n_parts):
        usage_metrics[f"part{p}"] = calc_usage_metrics(usage_counts[p])

    recon_mean = float(np.mean(recon_vals))
    commit_mean = float(np.mean(commit_vals))
    vel_mean = float(np.mean(vel_vals))
    training_style = maybe_run_training_style_eval(args, model, logger)

    metrics = {
        "config": vars(args),
        "loss": {
            "recon_mean": recon_mean,
            "commit_mean": commit_mean,
            "velocity_mean": vel_mean,
            "commit_over_recon": float(commit_mean / (recon_mean + 1e-8)),
            "perplexity_mean": float(np.mean(perplex_vals)),
            "perplexity_std": float(np.std(perplex_vals)),
        },
        "code_usage": usage_metrics,
        "code_sequence": {
            "num_sequences": int(codes_all.shape[0]),
            "unique_sequences": int(uniq),
            "duplicate_ratio": float(code_dup_ratio),
        },
        "nn_collapse_check": nn_metrics,
        "training_style_eval": training_style,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Saved metrics to %s", out_dir / "metrics.json")
    logger.info("Saved figures in %s", out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
