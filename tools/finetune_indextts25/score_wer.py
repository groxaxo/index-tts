#!/usr/bin/env python3
import argparse, json, re, unicodedata
from pathlib import Path
import whisper


def norm(s):
    s=unicodedata.normalize('NFKC',s).lower()
    s=re.sub(r'[^\wáéíóúüñ]+',' ',s,flags=re.UNICODE)
    return re.sub(r'\s+',' ',s).strip()


def edit(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--receipt',required=True); ap.add_argument('--model',default='medium'); ap.add_argument('--language',default='es'); ap.add_argument('--device')
    a=ap.parse_args(); p=Path(a.receipt); data=json.loads(p.read_text(encoding='utf-8')); model=whisper.load_model(a.model,device=a.device)
    rows=[]; we=ww=ce=cw=0
    for r in data['records']:
        if not r.get('ok'): rows.append({**r,'transcript':'','wer':1.0,'cer':1.0}); continue
        hyp=model.transcribe(r['wav'],language=a.language,fp16=(model.device.type=='cuda'))['text']
        ref=norm(r['text']); hn=norm(hyp); rw=ref.split(); hw=hn.split(); e=edit(rw,hw); c=edit(list(ref),list(hn))
        we+=e; ww+=len(rw); ce+=c; cw+=len(ref); row={**r,'transcript':hyp.strip(),'wer':e/max(1,len(rw)),'cer':c/max(1,len(ref))}; rows.append(row); print(json.dumps(row,ensure_ascii=False))
    out={'whisper_model':a.model,'language':a.language,'corpus_wer':we/max(1,ww),'corpus_cer':ce/max(1,cw),'word_edits':we,'reference_words':ww,'char_edits':ce,'reference_chars':cw,'records':rows}
    dst=p.with_name('wer.json'); dst.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(json.dumps({k:out[k] for k in ['corpus_wer','corpus_cer','word_edits','reference_words']},indent=2))

if __name__=='__main__': main()
