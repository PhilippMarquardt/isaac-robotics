import os, json, sys, numpy as np, torch
sys.path.insert(0,'/workspace/repos/gr00t-dreams')
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP

CKPT='/workspace/runs/idm_g1_pillow'
VAL="/workspace/data/split_80_20/val/G1_WB_Dex5_Pickup_Pillow"

cfg = DATA_CONFIG_MAP['g1_full_body']
ds = LeRobotSingleDataset(dataset_path=VAL, modality_configs=cfg.modality_config(),
    transforms=cfg.transform(), embodiment_tag=EmbodimentTag.G1_DEX5,
    video_backend='torchvision_av')
print('val len:', len(ds))

print('loading model...')
model = instantiate(OmegaConf.load('/workspace/repos/gr00t-dreams/IDM_dump/base.yaml'))
sd = load_file(f'{CKPT}/model.safetensors')
missing, unexp = model.load_state_dict(sd, strict=False)
print('missing:', len(missing), 'unexp:', len(unexp))
model = model.to('cuda').eval().to(torch.bfloat16)

meta = json.load(open(f'{CKPT}/experiment_cfg/metadata.json'))
act_meta = meta['g1_dex5']['modalities']['action']
order = ['left_leg','right_leg','waist','left_arm','right_arm','left_hand','right_hand']
slots = {}
off = 0
for k in order:
    d = act_meta[k]['shape'][0]
    slots[k] = (off, off+d); off += d
real_dim = off

stats = meta['g1_dex5']['statistics']['action']

# images/view_ids are already [V,...]; others get a batch dim
FLAT_KEYS = {'images','view_ids'}

def to_batch(s):
    out = {}
    for k,v in s.items():
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

errs_norm = {k: [] for k in order}
errs_phys = {k: [] for k in order}
N = min(200, len(ds))
rng = np.random.default_rng(0)
idxs = rng.choice(len(ds), N, replace=False)
for cnt,i in enumerate(idxs):
    s = ds[int(i)]
    batch = to_batch(s)
    gt = batch['actions'].float().cpu().numpy()[0]
    with torch.no_grad():
        out = model.get_action(batch)
    pred = out['action_pred'].float().cpu().numpy()[0]
    for k,(a,b) in slots.items():
        diff = (pred[:,a:b] - gt[:,a:b])
        errs_norm[k].append(float(np.sqrt((diff**2).mean())))
        sd_k = np.array(stats[k]['std'])
        errs_phys[k].append(float(np.sqrt(((diff*sd_k)**2).mean())))
    if (cnt+1) % 50 == 0:
        print(f'  {cnt+1}/{N}')

print(f'\n=== IDM eval — {N} val samples ===')
print(f'{"group":<12} {"RMSE(norm)":>12} {"RMSE(phys)":>12}')
for k in order:
    print(f'{k:<12} {np.mean(errs_norm[k]):>12.4f} {np.mean(errs_phys[k]):>12.4f}')
print(f'{"OVERALL":<12} {np.mean([v for vs in errs_norm.values() for v in vs]):>12.4f} {np.mean([v for vs in errs_phys.values() for v in vs]):>12.4f}')
