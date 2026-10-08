"""Summarise sweep.jsonl: how each hazard fared across the world. usage: python tools/sweep_report.py sweep.jsonl > report.md"""
import json, sys, collections, statistics as st


def cat(d):
    s = d.get("status", "")
    if s == "ERROR":
        return "ERROR"
    if s.startswith("FAILED"):
        return "verification FAILED"
    if "no model" in s:
        return "no model (no exposure / too little data)"
    if "small drift" in s:
        return "validated with small drift"
    if "hindcast FAILED" in s or "NO SKILL" in s or "no skill" in s.lower():
        return "hindcast failed / no skill"
    if "diagnosed" in s:
        return "validated after diagnosis"
    if "validated" in s:
        return "validated"
    if "verified only" in s:
        return "verified only"
    return "other: " + s[:40]


def main(path):
    D = [json.loads(l) for l in open(path)]
    sites = sorted({d["site"] for d in D})
    print(f"# Global Model Lab sweep\n\n{len(D)} studies at {len(sites)} sites, {len({d['hazard'] for d in D})} hazards.\n")
    by = collections.defaultdict(collections.Counter)
    for d in D:
        by[d["hazard"]][cat(d)] += 1
    cats = sorted({c for c in sum((list(v) for v in by.values()), [])})
    print("| hazard | n | " + " | ".join(cats) + " |\n|---|---|" + "---|" * len(cats))
    for h, c in by.items():
        print(f"| {h} | {sum(c.values())} | " + " | ".join(str(c.get(k, 0)) for k in cats) + " |")
    srcs = collections.Counter((d["hazard"], d.get("source")) for d in D if d.get("source"))
    if srcs:
        print("\nData source used by the extreme-value studies: " + ", ".join(f"{h}: {s} ×{n}" for (h, s), n in sorted(srcs.items())) + "\n")
    print("\n## Errors\n")
    for d in D:
        if d.get("status") == "ERROR":
            print(f"- {d['site']} / {d['hazard']}: {d.get('error')}")
    print("\n## Hindcast failures and no-skill results\n")
    for d in D:
        if cat(d) in ("hindcast failed / no skill", "verification FAILED"):
            print(f"- {d['site']} ({d['zone']}) / {d['hazard']}: {d['status']}  ks_p={d.get('ks_p')} drift={d.get('drift')} validation={d.get('validation')} {d.get('hc_reason') or ''}")
    for hz, key in (("drought", "r_anom"), ("river", "nse")):
        v = [(d["site"], d["validation"][key]) for d in D if d["hazard"] == hz and d.get("validation") and key in d["validation"]]
        if v:
            xs = [x for _, x in v]
            print(f"\n## {hz}: {key} across {len(v)} sites\n\nmedian {st.median(xs):.2f}, min {min(xs):.2f} ({min(v, key=lambda t: t[1])[0]}), max {max(xs):.2f}; below 0.5: {sum(1 for x in xs if x < 0.5)}\n")
    ev = [d for d in D if d["hazard"] == "earthquake" and d.get("b")]
    if ev:
        print(f"\n## earthquake: b-value across {len(ev)} sites: median {st.median(d['b'] for d in ev):.2f}, range {min(d['b'] for d in ev):.2f}–{max(d['b'] for d in ev):.2f}\n")
    t = [d for d in D if d["hazard"] in ("rain", "wind", "heat", "fire") and d.get("ks_p") is not None]
    if t:
        print(f"\n## extreme-value hindcast: {sum(1 for d in t if d.get('hc_passed'))} of {len(t)} passed outright (nominal false-fail rate ≈ 10 %)\n")
    print("\n## Slowest studies\n")
    for d in sorted(D, key=lambda d: -(d.get("wall_s") or 0))[:5]:
        print(f"- {d['site']} / {d['hazard']}: {d.get('wall_s')} s")


if __name__ == "__main__":
    main(sys.argv[1])
