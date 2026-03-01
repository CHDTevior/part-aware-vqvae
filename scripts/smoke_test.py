#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vqvae import HumanVQVAE


def build_args(partition_file: str | None, quantizer: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataname="t2m",
        quantizer=quantizer,
        partition_file=partition_file,
        mu=0.99,
        beta=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Part-Aware VQVAE smoke test")
    parser.add_argument(
        "--partition-file",
        type=str,
        default="partition_analysis/skeleton_partition.json",
        help="Path to skeleton partition json. Set empty string to skip.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--feat-dim", type=int, default=263)
    parser.add_argument("--nb-code", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument(
        "--quantizer",
        type=str,
        default="orig",
        choices=["orig", "ema", "ema_reset", "reset"],
        help="Use orig by default so CPU smoke test works everywhere.",
    )
    args = parser.parse_args()

    torch.manual_seed(123)

    device = torch.device(args.device)
    part_file = args.partition_file.strip() if args.partition_file else None

    if part_file:
        p = Path(part_file)
        if not p.exists():
            raise FileNotFoundError(f"partition file not found: {part_file}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if "partSeg" not in payload or not payload["partSeg"]:
            raise ValueError("partition file must contain non-empty key 'partSeg'")
        print(f"[OK] loaded partition file: {part_file} (n_parts={len(payload['partSeg'])})")
    else:
        print("[INFO] no partition file provided, will use fallback static partition")

    model_args = build_args(part_file, args.quantizer)
    model = HumanVQVAE(
        model_args,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.code_dim,
        down_t=2,
        stride_t=2,
        width=128,
        depth=2,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ).to(device)
    model.eval()

    x = torch.randn(args.batch_size, args.seq_len, args.feat_dim, device=device)

    with torch.no_grad():
        x_out, loss, perplexity = model(x, type="full")

        if x_out.shape != x.shape:
            raise RuntimeError(f"full forward shape mismatch: out={tuple(x_out.shape)} in={tuple(x.shape)}")
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is not finite: {loss.item()}")

        codes_flat = model(x, type="encode")  # (B, T_latent * n_parts)
        n_parts = model.vqvae.num_parts
        if codes_flat.shape[1] % n_parts != 0:
            raise RuntimeError(
                f"encode output cannot be split by n_parts: codes={tuple(codes_flat.shape)}, n_parts={n_parts}"
            )

        can_decode = all(hasattr(q, "forward_from_code_idx") for q in model.vqvae.quantizers)
        if can_decode:
            chunks = torch.chunk(codes_flat, n_parts, dim=1)
            codes = torch.stack(chunks, dim=-1)  # (B, T_latent, n_parts)
            x_dec = model(codes, type="decode")
        else:
            x_dec = None

    print(f"[OK] full forward: x -> {tuple(x_out.shape)}")
    print(f"[OK] encode output: {tuple(codes_flat.shape)}")
    if x_dec is not None:
        print(f"[OK] decode output: {tuple(x_dec.shape)}")
    else:
        print("[INFO] decode path skipped (quantizer has no forward_from_code_idx in current code)")
    print(f"[OK] loss={float(loss):.6f}, perplexity={float(perplexity):.4f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
