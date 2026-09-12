#!/usr/bin/env python3
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch, torchaudio
from indextts.infer_v2_5 import IndexTTS2
from train_lora import load_adapter


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--model-dir',default='checkpoints'); ap.add_argument('--config',default='checkpoints/config.yaml')
    ap.add_argument('--adapter'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--bf16',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--seeds',default='42,43,44'); ap.add_argument('--temperature',type=float,default=.8); ap.add_argument('--top-p',type=float,default=.8); ap.add_argument('--top-k',type=int,default=30)
    a=ap.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    suite=[json.loads(x) for x in Path(a.suite).read_text(encoding='utf-8').splitlines() if x.strip()]
    tts=IndexTTS2(cfg_path=a.config,model_dir=a.model_dir,use_bf16=a.bf16,device=a.device,use_cuda_kernel=False,use_qwen_emo=False)
    adapter_meta=None
    if a.adapter: adapter_meta=load_adapter(tts.gpt,a.adapter,dropout_override=0.0)
    records=[]
    for row in suite:
        for seed in [int(x) for x in a.seeds.split(',') if x.strip()]:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            rid=str(row['id']); wav=out/f'{rid}__seed{seed}.wav'; t0=time.perf_counter()
            result=tts.infer(row['speaker_ref'],row['text'],str(wav),row.get('lang','es'),temperature=a.temperature,top_p=a.top_p,top_k=a.top_k,verbose=False)
            elapsed=time.perf_counter()-t0
            ok=wav.is_file() and wav.stat().st_size>44
            dur=None; rtf=None
            if ok:
                info=torchaudio.info(str(wav)); dur=info.num_frames/info.sample_rate; rtf=elapsed/dur if dur else None
            rec={'id':rid,'seed':seed,'text':row['text'],'lang':row.get('lang','es'),'speaker_ref':row['speaker_ref'],'wav':str(wav),'ok':ok,'elapsed_s':round(elapsed,4),'audio_s':None if dur is None else round(dur,4),'rtf':None if rtf is None else round(rtf,4)}
            records.append(rec); print(json.dumps(rec,ensure_ascii=False))
    receipt={'suite':str(Path(a.suite).resolve()),'adapter':a.adapter,'adapter_meta':adapter_meta,'temperature':a.temperature,'top_p':a.top_p,'top_k':a.top_k,'records':records}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

if __name__=='__main__': main()
