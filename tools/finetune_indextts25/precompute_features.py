#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import torch, torchaudio
from indextts.infer_v2_5 import IndexTTS2, apply_pronunciation_annotations
from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize
from indextts.utils.tokenizer import lang_to_token


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def audio16(tts, path):
    wav,sr=tts._load_and_cut_audio(path,15,False)
    return torchaudio.transforms.Resample(sr,16000)(wav)


def w2v(tts, wav16):
    z=tts.extract_features(wav16,sampling_rate=16000,return_tensors='pt')
    return tts.get_emb(z['input_features'].to(tts.device),z['attention_mask'].to(tts.device))


def clean_text(tts,text,lang):
    text=tts.text_process.clean_pattern.sub(lambda x:tts.text_process.char_rep_map[x.group()],text)
    if lang in ['zh','zhen','en']: text=tts.text_process.normalize(text)
    elif lang in ['ja','es']: text=nemo_text_normalize(text,lang)
    if lang in ['ja','zh','zhen','en']: text=text.lower()
    if lang=='es': text=text.upper()
    text=apply_pronunciation_annotations(text)
    if lang=='ja': text=tts.ja_text_process.process_ja_text(text)
    return re.sub(r'<\|([^|]+)\|>',lambda m:f'<|{m.group(1).upper()}|>',text)


def campplus(tts,wav16):
    feat=torchaudio.compliance.kaldi.fbank(wav16.to(tts.device),num_mel_bins=80,dither=0,sample_frequency=16000)
    feat=feat-feat.mean(dim=0,keepdim=True)
    return tts.campplus_model(feat.unsqueeze(0))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--model-dir',default='checkpoints'); ap.add_argument('--config',default='checkpoints/config.yaml')
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--bf16',action='store_true'); ap.add_argument('--overwrite',action='store_true')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    tts=IndexTTS2(cfg_path=a.config,model_dir=a.model_dir,use_bf16=a.bf16,device=a.device,use_cuda_kernel=False,use_qwen_emo=False)
    receipt=[]
    for n,r in enumerate(rows(a.manifest)):
        rid=str(r.get('row_id') or f'{n:07d}'); dest=out/f'{rid}.pt'
        if dest.exists() and not a.overwrite:
            receipt.append({'row_id':rid,'file':str(dest),'status':'existing'}); continue
        lang=str(r.get('lang') or 'es').lower(); txt=clean_text(tts,str(r['text']),lang)
        # In IndexTTS 2.5 CAMPPlus mode language is injected through lang_embedding.
        # Do not prepend <|lang|> here or training would encode language twice.
        toks=tts.tokenizer.encode(txt,allowed_special='all')
        toks=torch.tensor(toks,dtype=torch.long)
        if len(toks)+2 > tts.gpt.text_pos_embedding.emb.num_embeddings:
            raise ValueError(f'{rid}: {len(toks)} text tokens exceed model capacity')
        target=w2v(tts,audio16(tts,r['audio']))
        codes=tts.get_scode(target).squeeze(0).long()
        if len(codes)+2 > tts.gpt.mel_pos_embedding.emb.num_embeddings:
            raise ValueError(f'{rid}: {len(codes)} semantic codes exceed model capacity')
        ref16=audio16(tts,r.get('ref_audio') or r['audio'])
        emo=w2v(tts,ref16).squeeze(0).to(torch.float32).cpu()
        spk=campplus(tts,ref16).squeeze(0).to(torch.float32).cpu()
        obj={
            'row_id':rid,'audio':r['audio'],'ref_audio':r.get('ref_audio') or r['audio'],'speaker':r['speaker'],
            'lang':lang,'lang_id':int(lang_to_token(lang)),'text':r['text'],'normalized_text':txt,
            'text_tokens':toks.cpu(),'mel_codes':codes.cpu(),'campplus':spk,'emo_condition':emo,
            'audio_sha256':r.get('audio_sha256'),'ref_is_target':bool(r.get('ref_is_target',False)),
        }
        torch.save(obj,dest)
        receipt.append({'row_id':rid,'file':str(dest),'text_tokens':len(toks),'mel_codes':len(codes),'status':'written'})
        print(f'[{n+1}] {rid}: text={len(toks)} codes={len(codes)}')
    (out/'feature_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

if __name__=='__main__': main()
