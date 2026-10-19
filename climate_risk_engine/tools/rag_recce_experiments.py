import json, numpy as np, collections, sys
sys.path.insert(0,'tools')
from cri.resident import casebase as CB, semantic as S
qs=json.load(open('tools/rag_bench_questions.json'))['paraphrase']
ix=CB._load(); mat=ix['sem'].mat
def evrank(ps,gold):
    seen=[]
    for p in ps:
        if p['event'] not in seen: seen.append(p['event'])
        if p['event'] in gold: return len(seen)
    return None
def metrics(fn):
    out={}
    for split in('p1','p2'):
        rs=[evrank(fn(q[split]),q['gold']) for q in qs]
        out[split]=(sum(r==1 for r in rs if r)/25, sum(1 for r in rs if r and r<=3)/25, sum(1/r for r in rs if r)/25)
    return out
# A: SEM_W sweep
for w in (0.0,0.35,0.5,0.7,0.9):
    CB.SEM_W=w
    m=metrics(lambda t:CB.retrieve(t,6)); print('SEM_W',w,{k:tuple(round(x,3) for x in v) for k,v in m.items()})
CB.SEM_W=0.35
# B: event-level dense (mean of passage vectors per event) as a pure ranker
evs=list(ix['events']); idx=collections.defaultdict(list)
for i,r in enumerate(ix['items']): idx[r['event']].append(i)
E=np.stack([mat[idx[e]].mean(0) for e in evs]); E/=np.linalg.norm(E,axis=1,keepdims=True)
# max-passage per event
def ev_dense(t,mode):
    v=S.embed_texts([t],query=True)[0]; v/=np.linalg.norm(v); s=mat@v
    if mode=='mean': sc=E@v
    else: sc=np.array([s[idx[e]].max() for e in evs])
    return [{'event':evs[i]} for i in sc.argsort()[::-1][:6]]
for mode in('mean','max'):
    m=metrics(lambda t:ev_dense(t,mode)); print('event-dense',mode,{k:tuple(round(x,3) for x in v) for k,v in m.items()})
print('--- fusion')
def pipe_ev(t):
    seen=[]
    for p in CB.retrieve(t,12):
        if p['event'] not in seen: seen.append(p['event'])
    return seen
def dense_ev(t,mode='max'):
    return [d['event'] for d in ev_dense(t,mode)] if False else None
def evlist(t,mode):
    v=S.embed_texts([t],query=True)[0]; v/=np.linalg.norm(v); s=mat@v
    sc=E@v if mode=='mean' else np.array([s[idx[e]].max() for e in evs])
    return [evs[i] for i in sc.argsort()[::-1][:12]]
def rrf(lists,ws,k=10):
    sc=collections.Counter()
    for l,w in zip(lists,ws):
        for r,e in enumerate(l,1): sc[e]+=w/(k+r)
    return [{'event':e} for e,_ in sc.most_common(6)]
for mode in('mean','max'):
  for wd in (0.5,1.0,1.5):
    m=metrics(lambda t:rrf([pipe_ev(t),evlist(t,mode)],[1.0,wd])); print('RRF pipe+dense',mode,'wd',wd,{k:tuple(round(x,3) for x in v) for k,v in m.items()})
print('--- p2 failures of event-dense max')
for q in qs:
    l=evlist(q['p2'],'max')[:3]
    if q['gold'][0] not in l[:1]: print(q['gold'],l,'|',q['p2'][:90])
