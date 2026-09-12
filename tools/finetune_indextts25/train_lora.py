#!/usr/bin/env python3
import argparse, json, math, random, re, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from omegaconf import OmegaConf
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint

DEFAULT_TARGET=r'^gpt\.h\.\d+\.(attn\.(c_attn|c_proj)|mlp\.(c_fc|c_proj))$'

class LoRAWeight(nn.Module):
    def __init__(self,shape,rank,alpha,dropout):
        super().__init__(); m,n=shape; self.rank=rank; self.scale=alpha/rank; self.dropout=dropout
        self.A=nn.Parameter(torch.empty(m,rank)); self.B=nn.Parameter(torch.zeros(rank,n))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
    def forward(self,w):
        a=F.dropout(self.A,p=self.dropout,training=self.training) if self.dropout else self.A
        return w+(a@self.B).to(w.dtype)*self.scale


def load_gpt(config,model_dir,device,bf16):
    cfg=OmegaConf.load(config); model=UnifiedVoice(**cfg.gpt,use_accel=False,spk_cond_mode='campplus')
    load_checkpoint(model,str(Path(model_dir)/cfg.gpt_checkpoint)); model.to(device)
    if bf16: model.bfloat16()
    return model,cfg


def inject(model,pattern,rank,alpha,dropout,targets=None):
    rx=re.compile(pattern); chosen=[]
    for name,module in model.named_modules():
        if targets is not None and name not in targets: continue
        if targets is None and not rx.match(name): continue
        w=getattr(module,'weight',None)
        if not isinstance(w,torch.nn.Parameter) or w.ndim!=2: continue
        parametrize.register_parametrization(module,'weight',LoRAWeight(tuple(w.shape),rank,alpha,dropout))
        chosen.append(name)
    if not chosen: raise RuntimeError(f'no 2-D weights matched {pattern}')
    return chosen


def adapter_state(model,targets,meta):
    sd={k:v.detach().cpu() for k,v in model.state_dict().items() if '.parametrizations.weight.0.' in k}
    return {'format':'indextts25-native-lora-v1','targets':targets,'state_dict':sd,**meta}


def load_adapter(model,path,dropout_override=None):
    a=torch.load(path,map_location='cpu'); meta=a
    inject(model,'$',int(a['rank']),float(a['alpha']),float(a.get('dropout',0) if dropout_override is None else dropout_override),targets=a['targets'])
    missing,unexpected=model.load_state_dict(a['state_dict'],strict=False)
    bad=[x for x in unexpected if 'parametrizations.weight.0.' in x]
    if bad: raise RuntimeError(f'adapter load failed: {bad}')
    return meta


def files(root,max_items=0):
    xs=sorted(Path(root).glob('*.pt')); return xs[:max_items] if max_items else xs


def one_loss(model,obj,device,bf16):
    text=obj['text_tokens'].to(device).long().unsqueeze(0); codes=obj['mel_codes'].to(device).long().unsqueeze(0)
    camp=obj['campplus'].to(device).unsqueeze(0); emo=obj['emo_condition'].to(device).unsqueeze(0)
    tl=torch.tensor([text.shape[1]],device=device); ml=torch.tensor([codes.shape[1]],device=device)
    el=torch.tensor([emo.shape[1]],device=device)
    with torch.amp.autocast('cuda',enabled=bf16 and str(device).startswith('cuda'),dtype=torch.bfloat16):
        hidden=model(camp,text,tl,codes,ml,emo,emo_cond_mel_lengths=el,do_spk_cond=True)
        logits=model.mel_head(hidden).float()
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),codes.reshape(-1))
    return loss

@torch.no_grad()
def evaluate(model,paths,device,bf16,limit=32):
    model.eval(); vals=[]
    for p in paths[:limit]: vals.append(float(one_loss(model,torch.load(p,map_location='cpu'),device,bf16)))
    model.train(); return sum(vals)/len(vals) if vals else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True); ap.add_argument('--dev-features')
    ap.add_argument('--config',default='checkpoints/config.yaml'); ap.add_argument('--model-dir',default='checkpoints')
    ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--bf16',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--rank',type=int,default=8); ap.add_argument('--alpha',type=float,default=16); ap.add_argument('--dropout',type=float,default=.05)
    ap.add_argument('--target-regex',default=DEFAULT_TARGET); ap.add_argument('--learning-rate',type=float,default=1e-5)
    ap.add_argument('--max-steps',type=int,default=500); ap.add_argument('--grad-accum',type=int,default=8); ap.add_argument('--warmup-steps',type=int,default=25)
    ap.add_argument('--weight-decay',type=float,default=0.0); ap.add_argument('--clip-grad',type=float,default=.5)
    ap.add_argument('--save-every',type=int,default=50); ap.add_argument('--eval-every',type=int,default=50); ap.add_argument('--seed',type=int,default=20260912)
    ap.add_argument('--max-items',type=int,default=0); ap.add_argument('--resume')
    a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    train=files(a.features,a.max_items); dev=files(a.dev_features) if a.dev_features else []
    if not train: raise SystemExit('no .pt feature files')
    model,cfg=load_gpt(a.config,a.model_dir,a.device,a.bf16)
    for p in model.parameters(): p.requires_grad=False
    if a.resume:
        meta=load_adapter(model,a.resume); targets=meta['targets']; start=int(meta.get('step',0))
    else:
        targets=inject(model,a.target_regex,a.rank,a.alpha,a.dropout); start=0
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError('no trainable LoRA parameters')
    opt=torch.optim.AdamW(trainable,lr=a.learning_rate,weight_decay=a.weight_decay)
    log=[]; order=list(train); random.Random(a.seed).shuffle(order); cursor=0
    model.train(); opt.zero_grad(set_to_none=True); wall=time.time()
    print(f'targets={len(targets)} trainable={sum(p.numel() for p in trainable):,} examples={len(train)}')
    for step in range(start+1,a.max_steps+1):
        acc=0.0
        for micro in range(a.grad_accum):
            if cursor>=len(order):
                random.Random(a.seed+step).shuffle(order); cursor=0
            obj=torch.load(order[cursor],map_location='cpu'); cursor+=1
            loss=one_loss(model,obj,a.device,a.bf16); (loss/a.grad_accum).backward(); acc+=float(loss.detach())
        if step==start+1:
            g=sum(float(p.grad.float().norm()) for p in trainable if p.grad is not None)
            if not math.isfinite(g) or g==0: raise RuntimeError(f'invalid LoRA gradient norm: {g}')
        grad=float(torch.nn.utils.clip_grad_norm_(trainable,a.clip_grad));
        scale=min(1.0,step/max(1,a.warmup_steps)); lr=a.learning_rate*scale
        for pg in opt.param_groups: pg['lr']=lr
        opt.step(); opt.zero_grad(set_to_none=True)
        rec={'step':step,'train_loss':acc/a.grad_accum,'grad_norm':grad,'lr':lr,'elapsed_s':round(time.time()-wall,2)}
        if not math.isfinite(rec['train_loss']): raise RuntimeError(f'non-finite loss at step {step}')
        if dev and (step==1 or step%a.eval_every==0): rec['dev_loss']=evaluate(model,dev,a.device,a.bf16)
        log.append(rec); print(json.dumps(rec))
        if step%a.save_every==0 or step==a.max_steps:
            meta={'step':step,'rank':a.rank,'alpha':a.alpha,'dropout':a.dropout,'target_regex':a.target_regex,'base_gpt_checkpoint':str(cfg.gpt_checkpoint),'seed':a.seed}
            path=out/f'adapter-step-{step:06d}.pt'; torch.save(adapter_state(model,targets,meta),path)
            # Fresh reload gate: instantiate base, inject adapter, and ensure state loads.
            fresh,_=load_gpt(a.config,a.model_dir,a.device,a.bf16); [p.requires_grad_(False) for p in fresh.parameters()]
            load_adapter(fresh,path); del fresh
            (out/'train_log.json').write_text(json.dumps(log,indent=2)+'\n')
    receipt={'base_gpt_checkpoint':str(cfg.gpt_checkpoint),'targets':targets,'trainable_parameters':sum(p.numel() for p in trainable),'seed':a.seed,'max_steps':a.max_steps,'grad_accum':a.grad_accum,'learning_rate':a.learning_rate,'rank':a.rank,'alpha':a.alpha,'dropout':a.dropout}
    (out/'run_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__': main()
