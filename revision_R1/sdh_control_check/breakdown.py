"""Break down per-direction SDH values by control type (final-token controls, no-op controls, others), with bootstrap CIs.

Run: python revision_R1/sdh_control_check/breakdown.py <rows.json>
"""
import json,statistics as st,random,sys
fin={('bat-lex-0','b_to_a'),('bat-lex-2','b_to_a'),('match-lex-0','b_to_a'),('match-lex-1','b_to_a'),('match-lex-2','b_to_a'),('match-lex-3','b_to_a'),('club-lex-1','a_to_b'),('club-lex-2','a_to_b'),('club-lex-2','b_to_a'),('club-lex-3','a_to_b')}
rows=json.load(open(sys.argv[1]))
def boot(v,n=2000,s=42):
    rng=random.Random(s); m=[st.mean(rng.choices(v,k=len(v))) for _ in range(n)]; m.sort(); return st.mean(v),m[int(.025*n)],m[int(.975*n)]
for name,sel in [('all',lambda r:True),('final-pos ctrl (10)',lambda r:(r['pair'],r['tag']) in fin),('noop (44)',lambda r:r['cls']=='noop'),('other (50)',lambda r:r['cls']!='noop' and (r['pair'],r['tag']) not in fin),('all minus final-pos (94)',lambda r:(r['pair'],r['tag']) not in fin)]:
    s=[r for r in rows if sel(r)]
    print('%-26s n=%3d E_ctrl %.3f [%.3f, %.3f]  E_target %.3f  delta %.3f [%.3f, %.3f]'%((name,len(s))+boot([r['ec'] for r in s])+(st.mean(r['et'] for r in s),)+boot([r['et']-r['ec'] for r in s])))
