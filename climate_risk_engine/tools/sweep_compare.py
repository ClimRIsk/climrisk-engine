"""Compare two sweeps of the same extreme-value hazards made with different data sources (e.g. NASA POWER vs ERA5). usage: sweep_compare.py a.jsonl b.jsonl"""
import json, sys, statistics as st


def load(p):
    return {(d["site"], d["hazard"]): d for d in (json.loads(l) for l in open(p))}


def main(a, b):
    A, B = load(a), load(b)
    print("# Cross-source comparison\n")
    for hz in ("rain", "heat", "wind", "fire"):
        rows = []
        for k in A:
            if k[1] == hz and k in B and A[k].get("levels") and B[k].get("levels") and A[k].get("source") != B[k].get("source"):
                la, lb = A[k]["levels"].get("50"), B[k]["levels"].get("50")
                if la and lb:
                    rows.append((k[0], la, lb, (la - lb) / lb))
        if not rows:
            print(f"- {hz}: no site pair with different sources yet\n"); continue
        r = [x[3] for x in rows]
        agree = sum(1 for x in rows if A[(x[0], hz)]["status"].split(";")[0] == B[(x[0], hz)]["status"].split(";")[0] and (("FAILED" in A[(x[0], hz)]["status"]) == ("FAILED" in B[(x[0], hz)]["status"])))
        print(f"- {hz}: {len(rows)} sites; 50-year level differs by median {st.median(r):+.0%} (range {min(r):+.0%} to {max(r):+.0%}); hindcast verdict (failed or not) agrees at {agree} of {len(rows)}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
