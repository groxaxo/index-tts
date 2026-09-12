#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from torch.nn.utils import parametrize
from train_lora import load_gpt, load_adapter


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='checkpoints/config.yaml'); ap.add_argument('--model-dir',default='checkpoints'); ap.add_argument('--adapter',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cpu')
    a=ap.parse_args(); model,cfg=load_gpt(a.config,a.model_dir,a.device,False); meta=load_adapter(model,a.adapter,dropout_override=0.0); model.eval()
    merged=[]
    mods=dict(model.named_modules())
    for name in meta['targets']:
        module=mods[name]
        if not parametrize.is_parametrized(module,'weight'): raise RuntimeError(f'{name} is not parametrized')
        parametrize.remove_parametrizations(module,'weight',leave_parametrized=True); merged.append(name)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    receipt={'format':'indextts25-merged-gpt-v1','base_gpt_checkpoint':str(cfg.gpt_checkpoint),'adapter':str(Path(a.adapter).resolve()),'adapter_step':meta.get('step'),'merged_targets':merged}
    torch.save({'model':model.state_dict(),'finetune_receipt':receipt},out)
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Fresh-load validation through the repository's normal checkpoint loader.
    fresh,_=load_gpt(a.config,a.model_dir,a.device,False)
    state=torch.load(out,map_location='cpu')['model']; missing,unexpected=fresh.load_state_dict(state,strict=False)
    if missing or unexpected: raise RuntimeError(f'merged reload failed: missing={missing[:5]} unexpected={unexpected[:5]}')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
