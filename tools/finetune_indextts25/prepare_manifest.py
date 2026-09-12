#!/usr/bin/env python3
import argparse, csv, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
import torchaudio


def sha256(path: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def load_rows(path: Path):
    if path.suffix.lower()=='.jsonl':
        return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=20260912)
    ap.add_argument('--train-frac', type=float, default=.8)
    ap.add_argument('--dev-frac', type=float, default=.1)
    a=ap.parse_args()
    src=Path(a.input); out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows=load_rows(src)
    if not rows: raise SystemExit('empty manifest')
    required={'audio','text','speaker'}
    cleaned=[]; by_spk=defaultdict(list)
    for i,r in enumerate(rows):
        miss=[k for k in required if not str(r.get(k,'')).strip()]
        if miss: raise ValueError(f'row {i}: missing {miss}')
        p=Path(r['audio']).expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        info=torchaudio.info(str(p)); dur=info.num_frames/info.sample_rate
        if dur<=0: raise ValueError(f'row {i}: zero duration')
        x=dict(r); x['audio']=str(p); x['lang']=str(x.get('lang') or 'es').lower()
        x['duration_s']=round(dur,6); x['audio_sha256']=sha256(p); x['row_id']=x.get('row_id') or f'{i:07d}'
        cleaned.append(x); by_spk[x['speaker']].append(x)
    for spk,items in by_spk.items():
        for j,x in enumerate(items):
            if not x.get('ref_audio'):
                ref=items[(j+1)%len(items)]['audio'] if len(items)>1 else x['audio']
                x['ref_audio']=ref
                x['ref_is_target']=(ref==x['audio'])
            else:
                rp=Path(x['ref_audio']).expanduser().resolve()
                if not rp.is_file(): raise FileNotFoundError(rp)
                x['ref_audio']=str(rp); x['ref_is_target']=(rp==Path(x['audio']))
    groups=defaultdict(list)
    for x in cleaned:
        session=str(x.get('session') or '').strip()
        key=f"{x['speaker']}::{session}" if session else f"{x['speaker']}::__speaker__"
        groups[key].append(x)
    keys=list(groups); random.Random(a.seed).shuffle(keys)
    total=sum(len(groups[k]) for k in keys); train_goal=total*a.train_frac; dev_goal=total*a.dev_frac
    splits={'train':[],'dev':[],'test':[]}
    for k in keys:
        if len(splits['train']) < train_goal: dst='train'
        elif len(splits['dev']) < dev_goal: dst='dev'
        else: dst='test'
        splits[dst].extend(groups[k])
    for name,rs in splits.items(): write_jsonl(out/f'{name}.jsonl', rs)
    receipt={
        'source':str(src.resolve()),'seed':a.seed,'rows':len(cleaned),
        'hours':round(sum(x['duration_s'] for x in cleaned)/3600,4),
        'speakers':len(by_spk),'groups':len(groups),
        'split_counts':{k:len(v) for k,v in splits.items()},
        'split_hours':{k:round(sum(x['duration_s'] for x in v)/3600,4) for k,v in splits.items()},
        'gender_counts':dict(Counter(str(x.get('gender','unknown')) for x in cleaned)),
        'dialect_counts':dict(Counter(str(x.get('dialect','unknown')) for x in cleaned)),
        'target_as_ref_count':sum(bool(x['ref_is_target']) for x in cleaned),
    }
    (out/'dataset_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
