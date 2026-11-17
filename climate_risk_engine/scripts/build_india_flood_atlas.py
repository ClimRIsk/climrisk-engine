"""Build src/cri/data/india_flood_atlas.json (+ india_districts_adm2.json.gz) from the NRSC/NDMA Flood Affected Area Atlas of India and geoBoundaries districts.

Inputs (download once; not committed):
  atlas.pdf  https://ndem.nrsc.gov.in/documents/downloads/Flood%20Affected%20Area%20%20Atlas%20of%20India%20-Satellite%20based%20study.pdf   (NRSC/ISRO for NDMA, Mar 2023, v1, 1998-2022)
  gb_ADM1.geojson, gb_ADM2.geojson   geoBoundaries gbOpen IND (2021 vintage, ODbL) _simplified.geojson
Usage: python scripts/build_india_flood_atlas.py atlas.pdf gb_ADM1.geojson gb_ADM2.geojson

Parse is checked: per-state district sums must equal the atlas's own TOTAL within rounding, else the state is flagged. Name matching atlas-district -> 2021 district is by normalised name
inside the same state; atlas districts are older than the 2021 boundaries (many districts have since been split), so unmatched names are kept and reported, never guessed.
"""
import difflib, gzip, json, re, subprocess, sys, unicodedata
from pathlib import Path

HDR = re.compile(r'Table\.?\s*(\d+)\.(\d+)\s*\.?\s*District[\s-]*wise\s+(?:statistics\s+of\s+)?flood\s+affected\s+area(?:s)?\s+(?:in|of)\s+(.+?)\s*(?:state|State)?\s*$', re.I)
ENT = re.compile(r'(?:(?<=\s)|^)(\d{1,3})\s+([A-Za-z][A-Za-z0-9 .&()\'/\-–]*?[A-Za-z)])\s{2,}(\d[\d,]*)(?=\s{2,}|\s*$)')
TOT = re.compile(r'TOTAL\s+(\d[\d,]*)', re.I)


def parse(pdf: str) -> tuple:
    txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout.split("\n")
    cur, data, totals = None, {}, {}
    for l in txt:
        m = HDR.search(l)
        if m:
            cur = m.group(3).strip(); data.setdefault(cur, {}); continue
        if cur is None:
            continue
        if "Glimpses of satellite" in l or re.search(r"Fig\.\s*\d+\.\d+", l):
            cur = None; continue
        t = TOT.search(l)
        if t:
            totals[cur] = int(t.group(1).replace(",", ""))
        for e in ENT.finditer(l):
            data[cur][int(e.group(1))] = (e.group(2).strip(), int(e.group(3).replace(",", "")))
    return {s: [v[k] for k in sorted(v)] for s, v in data.items()}, totals


def norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"\(.*?\)", "", s.lower())
    s = re.sub(r"[^a-z]", "", s)
    for a, b in (("kanchipuram", "kancheepuram"), ("tuticorin", "thoothukkudi"), ("nagappattinam", "nagapattinam"), ("bara banki", "barabanki")):
        s = s.replace(a.replace(" ", ""), b.replace(" ", ""))
    return s


def main(pdf, adm1, adm2):
    from shapely.geometry import shape, mapping
    from shapely.ops import transform
    import pyproj
    districts, totals = parse(pdf)
    check = {s: {"n": len(r), "sum": sum(x[1] for x in r), "atlas_total": totals.get(s)} for s, r in districts.items()}
    for s, c in check.items():
        c["ok"] = c["atlas_total"] is not None and abs(c["sum"] - c["atlas_total"]) <= max(10, 0.0005 * c["atlas_total"])
    st = [(f["properties"]["shapeName"], shape(f["geometry"])) for f in json.load(open(adm1))["features"]]
    proj = pyproj.Transformer.from_crs(4326, 6933, always_xy=True).transform
    D = []
    for f in json.load(open(adm2))["features"]:
        g = shape(f["geometry"]); p = g.representative_point()
        state = next((n for n, sg in st if sg.contains(p)), None)
        gs = g.simplify(0.004, preserve_topology=True)
        D.append({"name": f["properties"]["shapeName"], "state": state, "area_ha": round(transform(proj, g).area / 1e4), "geom": gs})
    alias = {"Jammu & Kashmir": "Jammu and Kashmir"}
    # spelling or renaming only (same district); anything that is a SPLIT or a merger is left unmatched on purpose
    ALIASES = {("Bihar", "Chhapra"): "Saran", ("Bihar", "Bhabhua"): "Kaimur (Bhabua)", ("Uttar Pradesh", "Badaun"): "Budaun", ("Maharashtra", "Garhchiroli"): "Gadchiroli",
               ("Maharashtra", "Raygad"): "Raigarh", ("Karnataka", "Bijapur / Vijayapura"): "Vijayapura", ("Odisha", "Deogarh"): "Debagarh", ("Odisha", "Sonapur"): "Subarnapur",
               ("Odisha", "Bauda"): "Boudh", ("Punjab", "Tam Taran"): "Tarn Taran", ("Punjab", "Nawan Shahr"): "Shahid Bhagat Singh Nagar",
               ("Punjab", "Sas Nagar"): "Sahibzada Ajit Singh Nagar", ("Madhya Pradesh", "East Nimar"): "Khandwa (East Nimar)", ("Telangana", "Bhadradri Kothagudem"): "Bhadradri",
               ("Assam", "Kamrup Metro"): "Kamrup Metropolitan"}
    rows, unmatched = [], []
    for s, r in districts.items():
        cand = [d for d in D if d["state"] and norm(d["state"]) == norm(alias.get(s, s))]
        for name, ha in r:
            best, br = None, 0.0
            al = ALIASES.get((s, name))
            for d in cand:
                if al and d["name"] == al:
                    best, br = d, 1.0
                    break
                a, b = norm(name), norm(d["name"])
                rr = 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()
                if rr > br: best, br = d, rr
            if best is not None and br >= 0.86:
                rows.append({"state": s, "district": name, "ha": ha, "matched": best["name"], "match_ratio": round(br, 2)})
            else:
                rows.append({"state": s, "district": name, "ha": ha, "matched": None, "match_ratio": round(br, 2)}); unmatched.append((s, name, ha))
    out = {"source": "NRSC/ISRO for NDMA, Flood Affected Area Atlas of India - Satellite based study, Version 1 (1998-2022), March 2023, Document NRSC-RSA-DMSG-FMHAD-MAR,2023-TR-2171-V1.0",
           "url": "https://ndem.nrsc.gov.in/documents/downloads/Flood%20Affected%20Area%20%20Atlas%20of%20India%20-Satellite%20based%20study.pdf",
           "licence": "NOT CHECKED: government publication; reuse terms not confirmed. District statistics only (no maps or imagery are reproduced).",
           "what": "Cumulative flood-affected area per district (ha) from multi-date optical and SAR satellite flood maps 1998-2022 plus CWC gauge levels. Includes rain-water accumulation in low land.",
           "limits": ["Atlas districts are older than the 2021 boundaries; a district that was split since may match its parent by name only.", "A district missing from a covered state has no mapped flood in the record; it is NOT shown as zero.",
                      "Cumulative extent over 25 years: no frequency, depth or return period."],
           "parse_check": check, "districts": rows}
    Path("src/cri/data/india_flood_atlas.json").write_text(json.dumps(out, indent=0))
    geo = [{"name": d["name"], "state": d["state"], "area_ha": d["area_ha"], "geom": mapping(d["geom"])} for d in D]
    for g in geo:                                            # 3 decimals is about 110 m, enough for a district lookup
        def rnd(c): return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 3), round(c[1], 3)]
        g["geom"]["coordinates"] = rnd(g["geom"]["coordinates"])
    with gzip.open("src/cri/data/india_districts_adm2.json.gz", "wt") as fh:
        json.dump({"source": "geoBoundaries gbOpen IND ADM2, 2021 (Pathways Data Pvt. Ltd., lgdirectory.gov.in), ODbL 1.0, simplified to ~400 m", "districts": geo}, fh, separators=(",", ":"))
    print("states", len(check), "bad", [s for s, c in check.items() if not c["ok"]], "rows", len(rows), "matched", sum(1 for r in rows if r["matched"]), "unmatched", len(unmatched))
    for u in unmatched[:60]: print("  unmatched", u)


if __name__ == "__main__":
    main(*sys.argv[1:4])
