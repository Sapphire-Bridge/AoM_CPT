"""Reimplementation of the archived SDH protocol at the fixed depth l* = round(0.25*(L-1)), with per-direction output.

Target patch and matched_token control patch are applied at l* for each of the 104 DISAMB
donor->receiver directions. The per-direction values (et = E_target, ec = E_ctrl, base margin,
control class) are written to a JSON file. The means reproduce Appendix Table D1 to four decimals
for GPT-2, Qwen2.5-0.5B and Llama-3.2-1B.

Run from the repository root:
    python revision_R1/sdh_control_check/sdh_reimplementation.py <model snapshot path> <gpt2|llama|qwen> <output.json>
Only the local paths were changed relative to the version that produced the archived rows.
"""
import sys, os, json, math, random, statistics as st, glob
os.environ["HF_HUB_OFFLINE"]="1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from selection_lib import span_for, select_matched, seed
torch.set_grad_enabled(False)
path=sys.argv[1]; kind=sys.argv[2]
tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).eval()
blocks = model.transformer.h if kind=="gpt2" else model.model.layers
L=len(blocks); layer=int(round(0.25*(L-1)))
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
items=[json.loads(l) for l in open(os.path.join(ROOT,"data_paper_hardened_v2","disamb_pairs.jsonl")) if l.strip()]

def enc(t): return tok(t,add_special_tokens=False)["input_ids"]
def score(prompt_ids, choices, patch=None):
    out={}
    for lab,conts in choices.items():
        vals=[]
        for c in conts:
            cid=enc(c); full=torch.tensor([prompt_ids+cid])
            h=None
            if patch is not None:
                pos,rep=patch
                def hook(m,i,o):
                    x=o[0] if isinstance(o,tuple) else o
                    x[:, pos, :]=rep
                    return o
                h=blocks[layer].register_forward_hook(hook)
            logits=model(input_ids=full,use_cache=False).logits[0].float()
            if h: h.remove()
            lp=torch.log_softmax(logits,-1)
            P=len(prompt_ids)
            s=sum(lp[P-1+j, cid[j]].item() for j in range(len(cid)))/len(cid)
            vals.append(s)
        t=torch.tensor(vals,dtype=torch.float64); out[lab]=float(torch.logsumexp(t,0)-math.log(len(vals)))
    return out
def margin(sc,exp): return sc[exp]-max(v for k,v in sc.items() if k!=exp)

rows=[]
for it in items:
    for di,(d,r) in enumerate(((it["a"],it["b"]),(it["b"],it["a"]))):
        tag="a_to_b" if di==0 else "b_to_a"
        ds,dt,dids=span_for(tok,d["prompt"],it["target"],it["target_occurrence"])
        rs,rt,rids=span_for(tok,r["prompt"],it["target"],it["target_occurrence"])
        exp=d["expected_label"]
        base=margin(score(rids,it["choices"]),exp)
        hs=model(input_ids=torch.tensor([dids]),output_hidden_states=True).hidden_states[layer+1][0]
        et=margin(score(rids,it["choices"],(rs,hs[ds,:])),exp)-base
        rng=random.Random(seed(it["pair_id"],tag,base_seed=0))
        pairs=select_matched(dids,rids,ds,rs,len(rs),2,2,8,rng)
        pairs=sorted(pairs,key=lambda qp:qp[1]); q=[a for a,b in pairs]; p=[b for a,b in pairs]
        ec=margin(score(rids,it["choices"],(p,hs[q,:])),exp)-base
        div=next((i for i in range(min(len(dids),len(rids))) if dids[i]!=rids[i]), min(len(dids),len(rids)))
        cls="noop" if all(a==b and b<div for a,b in pairs) else ("same_index_post_div" if all(a==b for a,b in pairs) else "pos_mismatch")
        rows.append(dict(pair=it["pair_id"],tag=tag,base=base,et=et,ec=ec,cls=cls,tgt_idx_match=(ds==rs),before=all(b<rs[0] for a,b in pairs)))
print("model",path.split("/")[-3] if "snapshots" in path else path,"L",L,"layer",layer,"n",len(rows))
print("E_target mean %.4f  E_ctrl mean %.4f  delta %.4f"%(st.mean(r["et"] for r in rows),st.mean(r["ec"] for r in rows),st.mean(r["et"]-r["ec"] for r in rows)))
for c in ["noop","pos_mismatch","same_index_post_div"]:
    s=[r for r in rows if r["cls"]==c]
    if s: print("  ctrl class %-20s n=%3d  mean E_ctrl %.4f  max|E_ctrl| %.4f  mean E_target %.4f"%(c,len(s),st.mean(r["ec"] for r in s),max(abs(r["ec"]) for r in s),st.mean(r["et"] for r in s)))
for m in [True,False]:
    s=[r for r in rows if r["tgt_idx_match"]==m]
    print("  target index match=%s n=%d mean E_target %.4f"%(m,len(s),st.mean(r["et"] for r in s)))
s=[r for r in rows if r["cls"]!="noop"]
print("  non-noop controls: n=%d mean E_ctrl %.4f mean delta %.4f"%(len(s),st.mean(r["ec"] for r in s),st.mean(r["et"]-r["ec"] for r in s)))
json.dump(rows,open(sys.argv[3],"w"))
