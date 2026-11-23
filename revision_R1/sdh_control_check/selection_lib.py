"""Standalone copy of the archived matched_token control selection and span lookup (used by sdh_reimplementation.py)."""
import json, sys, random, hashlib, collections, os
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
from transformers import AutoTokenizer


def seed(*parts, base_seed):
    h=hashlib.sha256(); h.update(str(int(base_seed)).encode())
    for p in parts: h.update(b"|"); h.update(str(p).encode())
    return int.from_bytes(h.digest()[:4],"little")

def excl(span,buffer,n):
    lo=max(0,min(span)-buffer); hi=min(n-1,max(span)+buffer); return set(range(lo,hi+1))

def find_nth(h,n_,occ):
    idx=-1; start=0
    for _ in range(occ+1):
        idx=h.find(n_,start)
        if idx<0: return -1
        start=idx+len(n_)
    return idx

def span_for(tok,text,sub,occ):
    ci=find_nth(text,sub,occ); ce=ci+len(sub)
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    sp=[i for i,(s,e) in enumerate(enc["offset_mapping"]) if s<ce and e>ci]
    return sp,[enc["input_ids"][i] for i in sp], enc["input_ids"]

def select_matched(donor_ids,recv_ids,donor_span,recv_span,k,buffer,donor_buffer,window,rng):
    center=int(round(sum(recv_span)/len(recv_span)))
    rex=excl(recv_span,buffer,len(recv_ids)); dex=excl(donor_span,donor_buffer,len(donor_ids))
    dpos={}
    for q,t in enumerate(donor_ids):
        if q in dex: continue
        dpos.setdefault(t,[]).append(q)
    elig_all=[p for p,t in enumerate(recv_ids) if p not in rex and t in dpos]
    if len(elig_all)<k: return None
    elig=elig_all; w=window; maxw=len(recv_ids)
    while True:
        inw=[p for p in elig_all if abs(p-center)<=w]
        if len(inw)>=k: elig=inw; break
        if w>=maxw: elig=elig_all; break
        w=min(maxw,max(w*2,w+1))
    rp=list(elig); rng.shuffle(rp); rp.sort(key=lambda p:abs(p-center)); rp=rp[:k]
    used=set(); pairs=[]
    for p in rp:
        cands=[q for q in dpos[recv_ids[p]] if q not in used] or dpos[recv_ids[p]]
        q=min(cands,key=lambda q:(abs(q-p),q)); used.add(q); pairs.append((q,p))
    return pairs

