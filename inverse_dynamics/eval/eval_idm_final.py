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
cfg = DATA_CONFIG_MAP['g1_full_body']

print('loading model...')
model = instantiate(OmegaConf.load('/workspace/repos/gr00t-dreams/IDM_dump/base.yaml'))
sd = load_file(f'{CKPT}/model.safetensors')
model.load_state_dict(sd, strict=False)
model = model.to('cuda').eval().to(torch.bfloat16)

meta = json.load(open(f'{CKPT}/experiment_cfg/metadata.json'))
act_meta = meta['g1_dex5']['modalities']['action']
order = ['left_leg','right_leg','waist','left_arm','right_arm','left_hand','right_hand']
slots={}; off=0
for k in order:
    d=act_meta[k]['shape'][0]; slots[k]=(off,off+d); off+=d
stats = meta['g1_dex5']['statistics']['action']

FLAT={'images','view_ids'}
def to_batch(s):
    out={}
    for k,v in s.items():
        if k=='embodiment_id': out[k]=torch.tensor([int(v)], device='cuda')
        elif isinstance(v, np.ndarray):
            t=torch.as_tensor(v).to('cuda'); out[k]=t if k in FLAT else t.unsqueeze(0)
        elif isinstance(v, list): out[k]=torch.as_tensor(v, device='cuda')
        else: out[k]=v
    for k in list(out):
        if isinstance(out[k], torch.Tensor) and out[k].dtype==torch.float32:
            out[k]=out[k].to(torch.bfloat16)
    return out

results={}
for split in ['train','val']:
    path=f'/workspace/data/split_80_20/{split}/G1_WB_Dex5_Pickup_Pillow'
    ds = LeRobotSingleDataset(dataset_path=path, modality_configs=cfg.modality_config(),
        transforms=cfg.transform(), embodiment_tag=EmbodimentTag.G1_DEX5, video_backend='torchvision_av')
    N=min(200,len(ds))
    rng=np.random.default_rng(0); idxs=rng.choice(len(ds), N, replace=False)
    errs_n={k:[] for k in order}; errs_p={k:[] for k in order}
    for cnt,i in enumerate(idxs):
        s=ds[int(i)]; batch=to_batch(s)
        gt=batch['actions'].float().cpu().numpy()[0]
        with torch.no_grad(): out=model.get_action(batch)
        pred=out['action_pred'].float().cpu().numpy()[0]
        for k,(a,b) in slots.items():
            diff=pred[:,a:b]-gt[:,a:b]
            errs_n[k].append(float(np.sqrt((diff**2).mean())))
            sd_k=np.array(stats[k]['std'])
            errs_p[k].append(float(np.sqrt(((diff*sd_k)**2).mean())))
    results[split]={
        'n_samples':N, 'dataset_size':len(ds),
        'rmse_normalized':{k:float(np.mean(v)) for k,v in errs_n.items()},
        'rmse_physical':{k:float(np.mean(v)) for k,v in errs_p.items()},
        'overall_rmse_normalized':float(np.mean([v for vs in errs_n.values() for v in vs])),
        'overall_rmse_physical':float(np.mean([v for vs in errs_p.values() for v in vs])),
    }
    print(split, results[split]['overall_rmse_normalized'], results[split]['overall_rmse_physical'])

with open('/workspace/idm_eval_summary.json','w') as f:
    json.dump(results, f, indent=2)
print('WROTE /workspace/idm_eval_summary.json')
