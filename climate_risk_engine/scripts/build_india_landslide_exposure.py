"""Parse Table 3 of the NRSC Landslide Atlas of India (2023) into src/cri/data/india_landslide_exposure.json.

Usage:  pdftotext -layout LandslideAtlas_2023.pdf ls.txt
        python scripts/build_india_landslide_exposure.py ls.txt

Table 3 ranks 147 districts (17 states, 2 UTs) by exposure to landslides in terms of socio-economic parameters (population, households, livestock, roads), normalised by district area.
It is an EXPOSURE rank, not a hazard count and not a probability. Districts outside the 147 were not ranked: that means "not assessed", never "safe".
Matching to the 2021 geoBoundaries districts is by accent-stripped name; renames are aliased by hand; districts split since are left unmatched on purpose.
"""
from __future__ import annotations

import difflib
import gzip
import json
import re
import sys
import unicodedata
from pathlib import Path

STATES = ["Uttaranchal", "Kerala", "J&K", "Sikkim", "Manipur", "Karnataka", "Karnatka", "Himachal Pradesh", "Assam", "Mizoram", "Tamil Nadu", "Meghalaya", "West Bengal",
          "Arunachal Pradesh", "Nagaland", "Tripura", "Goa", "Maharashtra"]
ENT = re.compile(r"(?<![\d])(\d{1,3})\s{2,}([A-Za-z][A-Za-z .&\-']*?)\s{2,}(" + "|".join(re.escape(s) for s in STATES) + r")(?=\s{2,}|\s*$)", re.I)
STATE_FIX = {"uttaranchal": "Uttarakhand", "j&k": "Jammu and Kashmir", "karnatka": "Karnataka"}
# renames / spelling only (same district). Splits (Karbi Anglong, Jaintia Hills) are NOT aliased.
ALIASES = {("Uttarakhand", "Dehra Dun"): "Dehradun", ("Uttarakhand", "Garhwal"): "Pauri Garhwal", ("Uttarakhand", "Naini Tal"): "Nainital",
           ("Jammu and Kashmir", "Rajauri"): "Rajouri", ("Jammu and Kashmir", "Anantanag"): "Anantnag", ("Jammu and Kashmir", "Leh-ladakh"): "Leh(Ladakh)",
           ("Jammu and Kashmir", "Poonch"): "Punch", ("Jammu and Kashmir", "Budgam"): "Badgam",
           ("Maharashtra", "Raygarh"): "Raigarh", ("Maharashtra", "Ahmadnagar"): "Ahmednagar", ("Maharashtra", "Mumbai"): "Mumbai City",
           ("Assam", "North Cachar Hills"): "Dima Hasao", ("Himachal Pradesh", "Lahul & Spiti"): "Lahul and Spiti", ("Tamil Nadu", "The Nilgiris"): "Nilgiris"}


def norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))
    return re.sub(r"[^a-z]", "", re.sub(r"\(.*?\)", "", s.lower().replace("&", "and")))


def parse(txt: str):
    lines = Path(txt).read_text(errors="ignore").splitlines()
    start = next(i for i, l in enumerate(lines) if "Table 3. Table shows the rank of districts of India for their exposure" in l)
    end = next(i for i, l in enumerate(lines) if i > start + 60 and "RESEARCH & DEVELOPMENT IN LANDSLIDES" in l)
    seen = {}
    for l in lines[start:end]:
        for m in ENT.finditer(l):
            r = int(m.group(1))
            if 1 <= r <= 147 and r not in seen:
                st = STATE_FIX.get(m.group(3).lower(), m.group(3).title() if m.group(3).islower() else m.group(3))
                seen[r] = (re.sub(r"\s+", " ", m.group(2).strip()), st)
    return seen


def main(txt: str, geo_gz: str = "src/cri/data/india_districts_adm2.json.gz"):
    seen = parse(txt)
    missing = [r for r in range(1, 148) if r not in seen]
    geo = json.loads(gzip.open(geo_gz, "rt").read())["districts"]
    rows, unmatched = [], []
    for r in sorted(seen):
        name, st = seen[r]
        states = {"jammuandkashmir": {"jammuandkashmir", "ladakh"}}.get(norm(st), {norm(st)})
        cand = [g for g in geo if g["state"] and norm(g["state"]) in states]
        al = ALIASES.get((st, name))
        best, br = None, 0.0
        for g in cand:
            if al and g["name"] == al:
                best, br = g, 1.0
                break
            a, b = norm(name), norm(g["name"])
            rr = 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()
            if rr > br:
                best, br = g, rr
        ok = best is not None and br >= 0.86
        rows.append({"rank": r, "district": name, "state": st, "matched": best["name"] if ok else None, "matched_state": best["state"] if ok else None, "match_ratio": round(br, 2)})
        if not ok:
            unmatched.append((r, st, name, round(br, 2), best["name"] if best else None))
    out = {"source": "NRSC/ISRO, Landslide Atlas of India: Mapping, Monitoring and R&D studies using Remote Sensing data, Feb 2023, Document NRSC-RSA-GSG-GMED-FEB 2023-TR-0002167-V1.0, Table 3",
           "url": "https://www.isro.gov.in/media_isro/pdf/LandslideAtlas_2023.pdf",
           "licence": "UNRESOLVED: NRSC's published licence (nrsc.gov.in/EOS_dissemination_Licensing) covers IRS images, not atlas PDFs; reuse terms for the atlas are not confirmed. District ranks only, with credit \"NRSC, ISRO\" (no maps or imagery are reproduced). Ask data@nrsc.gov.in before commercial reliance.",
           "what": "Rank of 147 districts (17 states, 2 UTs of the Himalaya, North-East and Western Ghats) by exposure to landslides in terms of socio-economic parameters (population, households, livestock, roads), from NRSC's ~80,000-landslide inventory 1998-2022. 1 = most exposed.",
           "limits": ["An exposure rank, not a hazard measure or a probability: it mixes where landslides are with how much is there to be hit.",
                      "Only 147 mountain districts were ranked. A district that is not in the list was NOT assessed; that is not the same as safe.",
                      "District scale. A site can be on stable ground in a high-rank district or on a slope in an unranked one.",
                      "Atlas districts are older than the 2021 boundaries; districts split since are left unmatched."],
           "parse_check": {"rows": len(seen), "expected": 147, "missing_ranks": missing}, "districts": rows}
    Path("src/cri/data/india_landslide_exposure.json").write_text(json.dumps(out, indent=0))
    print("rows", len(seen), "missing", missing, "matched", sum(1 for r in rows if r["matched"]), "unmatched", len(unmatched))
    for u in unmatched:
        print("  unmatched", u)


if __name__ == "__main__":
    main(*sys.argv[1:3])
