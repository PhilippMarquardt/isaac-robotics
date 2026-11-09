"""IDM trajectory generator from dreamer (Cosmos-LoRA) video.

Process:
  1. Load a dataset sample as a template (provides view_ids, vl_token_ids, etc.)
  2. Open cosmos mp4, decode + resize to 256x256, sample frames
  3. For each 16-frame window, replace 'images' with [frame_t, frame_t+16], run
     model.get_action, collect action_pred (16, 64)
  4. Concatenate windows -> (N, 64) action trajectory in NORMALIZED space
  5. Un-normalize per-group (using metadata stats) -> physical rad
  6. Save npz with both normalized and physical actions

Usage:
  python idm_gen_traj.py --video /path/to.mp4 --out /path/out.npz
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/repos/gr00t-dreams')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP


def load_video(path: str) -> np.ndarray:
    """Decode mp4 -> (N, 3, 256, 256) float32 in [0, 1]."""
    import torchvision
    frames, _, _ = torchvision.io.read_video(path, pts_unit='sec', output_format='TCHW')
    frames = frames.float() / 255.0  # (T, 3, H, W)
    # resize to 256x256
    frames = torch.nn.functional.interpolate(frames, size=(256, 256), mode='bilinear', align_corners=False)
    return frames.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ckpt', default='/workspace/runs/idm_g1_pillow')
    p.add_argument('--val_path', default='/workspace/data/split_80_20/val/G1_WB_Dex5_Pickup_Pillow')
    p.add_argument('--cfg_yaml', default='/workspace/repos/gr00t-dreams/IDM_dump/base.yaml')
    args = p.parse_args()

    frames = load_video(args.video)
    N = len(frames)
    print(f'video: {N} frames @ 256x256')

    cfg = DATA_CONFIG_MAP['g1_full_body']
    ds = LeRobotSingleDataset(
        dataset_path=args.val_path,
        modality_configs=cfg.modality_config(), transforms=cfg.transform(),
        embodiment_tag=EmbodimentTag.G1_DEX5, video_backend='torchvision_av',
    )
    template = ds[0]
    print('template keys:', list(template.keys()))

    print('loading model...')
    model = instantiate(OmegaConf.load(args.cfg_yaml))
    sd = load_file(f'{args.ckpt}/model.safetensors')
    missing, unexp = model.load_state_dict(sd, strict=False)
    print(f'missing={len(missing)} unexp={len(unexp)}')
    model = model.to('cuda').eval().to(torch.bfloat16)

    meta = json.load(open(f'{args.ckpt}/experiment_cfg/metadata.json'))
    act_meta = meta['g1_dex5']['modalities']['action']
    stats = meta['g1_dex5']['statistics']['action']
    order = ['left_leg', 'right_leg', 'waist', 'left_arm', 'right_arm', 'left_hand', 'right_hand']
    slots = {}
    off = 0
    for k in order:
        d = act_meta[k]['shape'][0]
        slots[k] = (off, off + d); off += d
    real_dim = off
    print(f'real action dim: {real_dim}')

    FLAT_KEYS = {'images', 'view_ids'}

    def to_batch(s):
        out = {}
        for k, v in s.items():
            if k == 'embodiment_id':
                out[k] = torch.tensor([int(v)], device='cuda')
            elif isinstance(v, np.ndarray):
                t = torch.as_tensor(v).to('cuda')
                out[k] = t if k in FLAT_KEYS else t.unsqueeze(0)
            elif isinstance(v, list):
                out[k] = torch.as_tensor(v, device='cuda')
            else:
                out[k] = v
        for k in list(out):
            if isinstance(out[k], torch.Tensor) and out[k].dtype == torch.float32:
                out[k] = out[k].to(torch.bfloat16)
        return out

    n_windows = (N - 1) // 16
    print(f'running {n_windows} windows...')
    all_preds_norm = []
    for w in range(n_windows):
        t0, t1 = w * 16, w * 16 + 16
        sample = dict(template)
        # IDM expects 2 frames [0, 16] per sample. Put cosmos [t0, t1] there.
        sample['images'] = np.stack([frames[t0], frames[t1]], axis=0)  # (2, 3, 256, 256)
        batch = to_batch(sample)
        with torch.no_grad():
            out = model.get_action(batch)
        pred = out['action_pred'].float().cpu().numpy()[0]  # (16, 64)
        all_preds_norm.append(pred[:, :real_dim])  # strip padding

    traj_norm = np.concatenate(all_preds_norm, axis=0)  # (n_windows*16, 41)
    print(f'trajectory shape: {traj_norm.shape}')

    # Un-normalize per group using min-max stats
    traj_phys = np.zeros_like(traj_norm)
    for k, (a, b) in slots.items():
        lo = np.array(stats[k]['q01'] if 'q01' in stats[k] else stats[k]['min'])
        hi = np.array(stats[k]['q99'] if 'q99' in stats[k] else stats[k]['max'])
        # min-max normalization was: x_n = 2*(x-lo)/(hi-lo) - 1
        # inverse: x = (x_n + 1)/2 * (hi-lo) + lo
        traj_phys[:, a:b] = (traj_norm[:, a:b] + 1) / 2 * (hi - lo) + lo

    np.savez(args.out, traj_norm=traj_norm, traj_phys=traj_phys,
             slots=np.array(list(slots.items()), dtype=object),
             video_path=args.video, n_frames=N)
    print(f'saved: {args.out}')
    print(f'phys range per group:')
    for k, (a, b) in slots.items():
        print(f'  {k:<12} min={traj_phys[:,a:b].min():.3f} max={traj_phys[:,a:b].max():.3f}')


if __name__ == '__main__':
    main()
