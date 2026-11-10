import os,sys,tempfile,random,collections,re
os.environ["CRI_DATA_DIR"]=tempfile.mkdtemp()
sys.path.insert(0,"tests")
import numpy as np
from cri.resident import nlu as N, corpus as C
m=N.get_model()
import nlu_battery as A, nlu_battery_b as B, nlu_battery_c as Cc
def rows(mod):
    return [(u,k) for k,v in mod.BATTERY.items() for u in v]
sets={"A":rows(A),"B":rows(B),"C":rows(Cc)}
allr=[]
for k,v in sets.items(): print(k,len(v)); allr+=v
def ok(pred,true):
    if true in("offtopic","unknown"): return pred=="unknown"
    return pred==true
# calibration
bk=collections.defaultdict(lambda:[0,0])
hiwrong=[]
for u,t in allr:
    p=m.predict(u); c=p["confidence"]; b=min(int(c*5),4)
    h=ok(p["intent"],t); bk[b][0]+=h; bk[b][1]+=1
    if not h and p["intent"]!="unknown" and c>=0.6: hiwrong.append((u,t,p["intent"],c))
print("calibration (conf bucket: acc, n)")
for b in sorted(bk): print(f"  {b*0.2:.1f}-{b*0.2+0.2:.1f}: {bk[b][0]/bk[b][1]:.2f} n={bk[b][1]}")
print("wrong with conf>=0.6:",len(hiwrong)); [print("  ",x) for x in hiwrong[:8]]
# typo robustness
random.seed(1)
def typo(s,k):
    s=list(s)
    for _ in range(k):
        i=random.randrange(len(s))
        if s[i].isalpha():
            op=random.choice("dsr")
            if op=="d": s[i]=""
            elif op=="s" and i+1<len(s): s[i],s[i+1]=s[i+1],s[i]
            else: s[i]=random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(s)
ind=[(u,t) for u,t in allr if t not in("offtopic","unknown")]
for k in (0,1,2,3):
    c=w=a=0
    for u,t in ind:
        p=m.predict(typo(u,k) if k else u)["intent"]
        c+=p==t; a+=p=="unknown"; w+=(p!=t and p!="unknown")
    print(f"typos={k}: correct {c/len(ind):.3f} asked {a/len(ind):.3f} wrong-tool {w/len(ind):.3f} (n={len(ind)})")
# case/punct
for name,f in (("UPPER",str.upper),("no-space-punct",lambda s:re.sub(r"[^\w\s]","",s)),("prefix 'hey quick one, '",lambda s:"hey quick one, "+s),("suffix ' thanks!'",lambda s:s+" thanks!"),("prefix 'please can you '",lambda s:"please can you "+s)):
    c=w=0
    for u,t in ind:
        p=m.predict(f(u))["intent"]; c+=p==t; w+=(p!=t and p!="unknown")
    print(f"{name}: correct {c/len(ind):.3f} wrong-tool {w/len(ind):.3f}")
# hybrid vs model-only
c1=sum(m.predict(u,hybrid=False)["intent"]==t for u,t in ind)/len(ind)
print("model-only (no keyword cues):",round(c1,3))
# corpus balance & collisions
rs=C.utterances(); cnt=collections.Counter(i for u,i,t in rs); print("train per class:",sorted(cnt.items(),key=lambda x:x[1]))
tp=collections.Counter((i,t) for u,i,t in rs); print("templates per class:",sorted(collections.Counter(i for i,t in tp).items(),key=lambda x:x[1])[:8])
feats=set()
for u,i,t in rs: feats.update(N._feats(u))
hs=collections.Counter(N._h(f) for f in feats)
print("distinct features",len(feats),"buckets used",len(hs),"features sharing a bucket",sum(v for v in hs.values() if v>1), "of", len(feats))
# vocabulary: domain terms never in training
voc=set()
for u,i,t in rs: voc.update(re.findall(r"[a-z0-9]+",u.lower()))
terms="sfdr pai pcaf esrs csrd tcfd issb taxonomy insurance premium deductible wildfire drought heat hail tornado landslide earthquake permafrost biodiversity tnfd water stress credit pd lgd ecl merton paris alignment sbti abatement macc methane fueleu cii aer eexi ets corsia registry offset credits cea egrid residual mix desnz ember ngfs ssp rcp cmip downscaling utci wbgt plume portfolio var ebitda dcf stranded sensitivity montecarlo oasis supplier scope3".split()
print("domain terms absent from training vocabulary:",[t for t in terms if t not in voc])
