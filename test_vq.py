# test_vqvae_main.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

# === 你的工程内的导入 ===
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer

from motion_dataset import Text2MotionDataset
from utils_motion.train_opt import TrainT2MOptions

from exit.utils import generate_src_mask  # 未用到，但保留


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  # 定义一个 opt 字典
  data_root = '/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/humanml3d/HumanML3D'
  opt = {
      "max_motion_length": 210,       # int
      "dataset_name": "t2m",          # str, 可选 "t2m" 或其他
      "motion_dir": os.path.join(data_root, 'new_joint_vecs'),      # 存放 .npy 的路径
      "text_dir": os.path.join(data_root, 'texts'),          # 存放 .txt 的路径
      "unit_length": 1
  }
  from argparse import Namespace
  opt = Namespace(**opt)

  mean = np.load('/scratch/ts1v23/workspace/mogo_developer/checkpoints/t2m/rvq_n8192_d128/meta/mean.npy')
  std = np.load('/scratch/ts1v23/workspace/mogo_developer/checkpoints/t2m/rvq_n8192_d128/meta/std.npy')

  train_split_file = os.path.join('/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/humanml3d/HumanML3D', 'train.txt')
  val_split_file = os.path.join('/scratch/ts1v23/workspace/motion-latent-diffusion-main/datasets/humanml3d/HumanML3D', 'val.txt')

  # train_set = Text2MotionDataset(opt, mean, std, train_split_file)
  train_set = Text2MotionDataset(opt, mean, std, val_split_file)
  val_set = Text2MotionDataset(opt, mean, std, val_split_file)

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=not config.data.streaming,
    persistent_workers=True)
  
  
  # generator = torch.Generator().manual_seed(valid_seed)
  valid_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False)
    # Will be used in generative perplexity calculation
  
  return train_loader, valid_loader

# ====== 你给的模型定义（略有小修：不改逻辑，仅加 type hints / 设备选择） ======
class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        output_dim = 251 if args.dataname == 'kit' else 263

        self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t,
                               width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)

        self.decoder = Decoder(output_dim, output_emb_width, down_t, stride_t,
                               width, depth, dilation_growth_rate,
                               activation=activation, norm=norm)

        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        else:
            raise ValueError(f"Unknown quantizer: {args.quantizer}")

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        return x.permute(0, 2, 1)

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, code_dim,
                               down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def forward(self, x, type='full'):
        if type == 'full':
            x_out, loss, perplexity = self.vqvae(x)
            return x_out, loss, perplexity
        elif type == 'encode':
            quants = self.vqvae.encode(x)  # (B, T')
            return quants
        elif type == 'decode':
            x_out = self.vqvae.forward_decoder(x)
            return x_out
        else:
            raise ValueError(f'Unknown "{type}" type')

# ================== 实用函数 ==================
def strip_module_prefix(state):
    """去掉 DataParallel 的 'module.' 前缀"""
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state

def guess_state_dict(ckpt):
    """尽量从 checkpoint 中取出 state_dict"""
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "net", "netG", "generator"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    # 直接就是 state_dict
    return ckpt

def build_args_from_ckpt(ckpt, cli_args):
    """优先用 ckpt 里的 args 字段来还原训练配置；没有就用命令行"""
    class Args:
        pass
    a = Args()

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    a.dataname  = ckpt_args.get("dataname", cli_args.dataname)
    a.quantizer = ckpt_args.get("quantizer", cli_args.quantizer)
    a.nb_code   = ckpt_args.get("nb_code", cli_args.nb_code)
    a.code_dim  = ckpt_args.get("code_dim", cli_args.code_dim)
    # 下面这些一般变化小，保守用 CLI
    a.output_emb_width = cli_args.output_emb_width
    a.down_t = cli_args.down_t
    a.stride_t = cli_args.stride_t
    a.width = cli_args.width
    a.depth = cli_args.depth
    a.dilation_growth_rate = cli_args.dilation_growth_rate
    a.activation = cli_args.activation
    a.norm = cli_args.norm
    return a

def make_dummy_batch(batch_size, seq_len, dataname, device):
    feat_dim = 251 if dataname == "kit" else 263
    # 构造一个平滑的随机序列，便于网络重构
    x = torch.randn(batch_size, seq_len, feat_dim, device=device)
    x = torch.cumsum(x, dim=1) / 10.0
    return x

# ================== 主流程 ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/scratch/ts1v23/workspace/MMM/output/vq/2024-06-03-20-22-07_retrain/net_last.pth")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=196)

    # 如果 ckpt 里没有 args，就用这些兜底
    parser.add_argument("--dataname", type=str, default="t2m", choices=["kit", "t2m"])
    parser.add_argument("--quantizer", type=str, default="ema_reset", choices=["ema_reset", "orig", "ema", "reset"])
    parser.add_argument("--nb_code", type=int, default=8192)
    parser.add_argument("--code_dim", type=int, default=32)

    # 结构参数（通常与训练一致；如有偏差，load_state_dict 会用 strict=False 兼容）
    parser.add_argument("--output_emb_width", type=int, default=512)
    parser.add_argument("--down_t", type=int, default=3)
    parser.add_argument("--stride_t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation_growth_rate", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--norm", type=str, default=None)

    parser.add_argument("--outdir", type=str, default="./vqvae_smoketest")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[INFO] loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = guess_state_dict(ckpt)
    state = strip_module_prefix(state)

    # 用 ckpt 里的 args（如存在）覆盖 CLI 中的关键项
    hydra_args = build_args_from_ckpt(ckpt, args)

    print(f"[INFO] inferred dataname={hydra_args.dataname}, quantizer={hydra_args.quantizer}, "
          f"nb_code={getattr(hydra_args,'nb_code','?')}, code_dim={getattr(hydra_args,'code_dim','?')}")

    # 构建模型
    model = HumanVQVAE(
        hydra_args,
        nb_code=8192,
        code_dim=32,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.activation,
        norm=args.norm
    ).to(device)

    # 加载权重（允许非严格以兼容少量名不对的 buffer/键）
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    model.eval()

    # 构造一批假数据，跑 full → encode → decode
    with torch.no_grad():
        x = make_dummy_batch(args.batch_size, args.seq_len, hydra_args.dataname, device)
        x_out, loss, perplexity = model(x, type='full')
        mse = torch.mean((x_out - x) ** 2).item()

        print(f"[RESULT] forward: mse={mse:.6f}, perplexity={float(perplexity):.3f}, loss={float(loss):.6f}")
        np.save(os.path.join(args.outdir, "recon_input.npy"), x.detach().cpu().numpy())
        np.save(os.path.join(args.outdir, "recon_output.npy"), x_out.detach().cpu().numpy())

        # encode时候

        # 输入的X的shapetorch.Size([2, 192, 263])
        # 输出的quants的shapetorch.Size([2, 48])
        print(f"输入的X的shape{x.shape}")
        quants = model(x, type='encode')  # (B, T') 这里的 T' 取决于时域下采样
        print(f"输出的quants的shape{quants.shape}")
        print(f"[RESULT] encode: quants.shape={tuple(quants.shape)}, "
              f"min={int(quants.min())}, max={int(quants.max())}")
        np.save(os.path.join(args.outdir, "quants.npy"), quants.detach().cpu().numpy())

        # decode（把量化 index 解码回去）
        x_dec = model(quants, type='decode')
        mse_dec = torch.mean((x_dec - x) ** 2).item()
        print(f"[RESULT] decode: mse_dec={mse_dec:.6f}")
        np.save(os.path.join(args.outdir, "decode_output.npy"), x_dec.detach().cpu().numpy())

    print(f"[DONE] artifacts saved to: {args.outdir}")

if __name__ == "__main__":
    main()



'''
python -u test_vq.py \
  --ckpt /scratch/ts1v23/workspace/MMM/output/vq/2024-06-03-20-22-07_retrain/net_last.pth \
  --device cuda:0 \
  --batch_size 2 \
  --seq_len 192 \
  --dataname t2m \
  --quantizer ema_reset \
  --code_dim 32 \
  --nb_code 8192 \
  --down_t 2 \
  --stride_t 2 \
  --width 512 \
  --depth 3 \
  --dilation_growth_rate 3 \
  --activation relu
'''