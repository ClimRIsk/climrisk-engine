"""Global Model Lab sweep: run every hazard study at ~36 sites spanning every climate zone and continent, record what the engine established at each,
and summarise pass rates and failure modes. Resumable: results are appended to sweep.jsonl.  usage: python tools/global_sweep.py [out_dir] [hazard,hazard] """
import json, os, sys, time, traceback

SITES = [  # name, lat, lon, zone
    ("Dhaka", 23.81, 90.41, "tropical monsoon"), ("Singapore", 1.35, 103.82, "tropical rainforest"), ("Manaus", -3.12, -60.02, "tropical rainforest"), ("Lagos", 6.52, 3.38, "tropical coastal"), ("Mumbai", 19.08, 72.88, "tropical monsoon"),
    ("Nairobi", -1.29, 36.82, "tropical highland savanna"), ("Dakar", 14.69, -17.45, "semi-arid coast"), ("Riyadh", 24.71, 46.68, "hot desert"), ("Phoenix", 33.45, -112.07, "hot desert"), ("Alice Springs", -23.70, 133.88, "hot desert"),
    ("Miami", 25.76, -80.19, "humid subtropical"), ("Hong Kong", 22.32, 114.17, "humid subtropical"), ("Sao Paulo", -23.55, -46.63, "humid subtropical"), ("Shanghai", 31.23, 121.47, "humid subtropical"), ("Houston", 29.76, -95.37, "humid subtropical"),
    ("Sydney", -33.87, 151.21, "mediterranean/oceanic"), ("Los Angeles", 34.05, -118.24, "mediterranean"), ("Athens", 37.98, 23.73, "mediterranean"), ("Cape Town", -33.92, 18.42, "mediterranean"), ("Santiago", -33.45, -70.67, "mediterranean"),
    ("London", 51.51, -0.13, "temperate oceanic"), ("Berlin", 52.52, 13.40, "temperate"), ("New York", 40.71, -74.00, "temperate continental"), ("Tokyo", 35.68, 139.69, "humid subtropical/temperate"), ("Mexico City", 19.43, -99.13, "subtropical highland"),
    ("Moscow", 55.76, 37.62, "cold continental"), ("Toronto", 43.65, -79.38, "cold continental"), ("Ulaanbaatar", 47.89, 106.91, "cold semi-arid"), ("Anchorage", 61.22, -149.90, "subarctic"), ("Reykjavik", 64.15, -21.94, "subpolar oceanic"),
    ("Kathmandu", 27.72, 85.32, "mountain"), ("La Paz", -16.50, -68.15, "high altitude"), ("Addis Ababa", 9.03, 38.74, "tropical highland"),
    ("Male", 4.18, 73.51, "small island"), ("Honolulu", 21.31, -157.86, "island"), ("Auckland", -36.85, 174.76, "temperate oceanic"), ("Manila", 14.60, 120.98, "tropical coastal"),
]
HAZ = ["rain", "wind", "drought", "river", "fire", "heat", "cyclone", "earthquake"]


def brief(r: dict) -> dict:
    n = r.get("numbers", {}) or {}
    out = {"status": r["status"], "ok": r["ok"], "elapsed": r.get("elapsed_s"), "nverif": len(r.get("verification", [])), "nverif_fail": sum(1 for v in r.get("verification", []) if not v["ok"])}
    hc = n.get("hindcast")
    if isinstance(hc, dict):
        out["hc_passed"] = hc.get("passed"); out["ks_p"] = hc.get("ks_p"); out["hc_reason"] = hc.get("reason") or hc.get("bias_note")
        if hc.get("exceedances"):
            out["exceed_p"] = [e["p"] for e in hc["exceedances"]]
    elif isinstance(hc, list):
        out["hc_p"] = [h.get("p") for h in hc]
    if n.get("drift50") is not None:
        out["drift"] = n["drift50"]
    if "trend" in n:
        out["trend_sig"] = n["trend"].get("significant"); out["trend_slope_dec"] = n["trend"].get("slope_per_decade")
    if "levels" in n and n["levels"]:
        out["levels"] = {str(k): v for k, v in n["levels"].items()}
    v = n.get("validation")
    if isinstance(v, dict):
        out["validation"] = {k: v[k] for k in ("r", "r_anom", "nse", "skill_vs_clim", "rmse", "n_days") if k in v}
    for k in ("n_storms", "b", "mc", "p_b", "ks_p", "rates"):
        if k in n:
            out[k] = n[k]
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else HAZ
    shard, nshard = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sweep.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line); done.add((d["site"], d["hazard"]))
            except Exception:
                pass
    from cri.modellab import hazards
    for si, (name, lat, lon, zone) in enumerate(SITES):
        if si % nshard != shard:
            continue
        for hz in only:
            if (name, hz) in done:
                continue
            t0 = time.time()
            rec = {"site": name, "lat": lat, "lon": lon, "zone": zone, "hazard": hz}
            try:
                r = hazards.run_generic(hz, lat, lon, name, {})
                rec.update(brief(r))
            except Exception as e:
                rec.update({"status": "ERROR", "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            rec["wall_s"] = round(time.time() - t0, 1)
            with open(path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(f'{name:14s} {hz:10s} {rec["status"][:70]:70s} {rec["wall_s"]:6.1f}s', flush=True)


if __name__ == "__main__":
    main()
