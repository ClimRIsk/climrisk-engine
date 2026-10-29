"""Run the question batteries offline (no network) and print every miss. Usage: python tools/run_battery.py [ship|build]"""
import sys
sys.path[:0] = ["src", "tests"]
import battery_cases as B
from cri.maritime import answer as A
from cri.resident import feasibility as FE


def close(a, b):
    return abs(float(a) - float(b)) < 1e-6 if isinstance(b, (int, float)) and not isinstance(b, bool) and isinstance(a, (int, float)) else str(a).lower() == str(b).lower()


def ship():
    miss = 0
    for text, exp in B.SHIP:
        errs = []
        if not A.matches(text):
            errs.append("NOT ROUTED")
        p = A.parse(text)
        for k, v in exp.items():
            if k in ("bunker_usd_per_t", "charter_usd_per_day", "eua_price"):
                got = (p.get("market") or {}).get(k)
            else:
                got = p.get(k)
            if got is None or not close(got, v):
                errs.append(f"{k}: want {v!r} got {got!r}")
        if errs:
            miss += 1
            print("MISS:", text[:90], "|", "; ".join(errs))
    for text in B.SHIP_NEGATIVE:
        if A.matches(text):
            miss += 1
            print("FALSE ROUTE:", text)
    n = len(B.SHIP) + len(B.SHIP_NEGATIVE)
    print(f"ship: {n - miss}/{n} cases fully right")


def build():
    miss = 0
    for text, exp in B.BUILD:
        errs = []
        if not FE.matches(text):
            errs.append("NOT ROUTED")
        p = FE.parse(text) if hasattr(FE, "parse") else {"storeys": FE.storeys(text)}
        for k, v in exp.items():
            got = p.get(k)
            if got is None or not close(got, v):
                errs.append(f"{k}: want {v!r} got {got!r}")
        if errs:
            miss += 1
            print("MISS:", text[:90], "|", "; ".join(errs))
    for text in B.BUILD_NEGATIVE:
        if FE.matches(text):
            miss += 1
            print("FALSE ROUTE:", text)
    n = len(B.BUILD) + len(B.BUILD_NEGATIVE)
    print(f"build: {n - miss}/{n} cases fully right")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ship", "all"):
        ship()
    if which in ("build", "all"):
        build()
