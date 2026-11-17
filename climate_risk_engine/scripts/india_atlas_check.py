import time, json
from concurrent.futures import ThreadPoolExecutor
from cri import siteprofile as SP
from cri.analysis import india_flood_atlas as A
CITIES={"Patna":(25.5941,85.1376),"Gorakhpur":(26.76,83.37),"Guwahati":(26.1445,91.7362),"Kolkata":(22.5726,88.3639),"Bhubaneswar":(20.2961,85.8245),"Surat":(21.1702,72.8311),
"Ahmedabad":(23.0225,72.5714),"Hyderabad":(17.385,78.4867),"Vijayawada":(16.5062,80.648),"Chennai-Velachery":(12.9815,80.218),"Madurai":(9.9252,78.1198),"Kochi":(9.9312,76.2673),
"Bengaluru":(12.9716,77.5946),"Pune":(18.5204,73.8567),"Nagpur":(21.1458,79.0882),"Lucknow":(26.8467,80.9462),"Jaipur":(26.9124,75.7873),"Dehradun":(30.3165,78.0322),
"Bhopal":(23.2599,77.4126),"Raipur":(21.2514,81.6296),"Ranchi":(23.3441,85.3096),"Muzaffarpur":(26.1209,85.3647),"Darbhanga":(26.1542,85.8918),"Cuttack":(20.4625,85.883),"Varanasi":(25.3176,82.9739)}
def one(item):
    n,(la,lo)=item; at=A.lookup(la,lo); t=time.time()
    try:
        p=SP.profile(la,lo,n,only=["river_flood"],budget_s=60); e=p["hazards"]["river_flood"]
        return n,at.get("status"),at.get("atlas_ha"),at.get("national_percentile"),e["status"],e["severity"],e.get("value"),round(time.time()-t)
    except Exception as ex: return n,at.get("status"),at.get("atlas_ha"),at.get("national_percentile"),"err",None,None,str(ex)[:60]
with ThreadPoolExecutor(4) as ex: R=list(ex.map(one,CITIES.items()))
json.dump(R,open('/tmp/probe/atlascheck.json','w'))
for r in R: print(r)
