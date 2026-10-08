"""One-time setup: python -m cri.maintainer.setup  — sets your PIN (stored only as a salted hash) and the repositories the agent may push."""
import getpass
import sys

from . import push

DEFAULT = ["~/Documents/Claude/Projects/climrisk-engine-space/climate_risk_engine", "~/Documents/Claude/Projects/Climate Financial Risk Modelling"]


def main() -> None:
    print("Maintainer setup (private to this computer). Repositories the agent may push:")
    repos = [r for r in (sys.argv[1:] or DEFAULT)]
    for r in repos:
        print("  ", r)
    p1 = getpass.getpass("Choose a PIN (8+ characters): "); p2 = getpass.getpass("Repeat: ")
    if p1 != p2 or len(p1) < 8:
        raise SystemExit("PINs differ or are shorter than 8 characters; nothing saved.")
    push.save_config(p1, repos)
    ext = push.HOME / "extensions"
    have = ext.read_text().splitlines() if ext.exists() else []
    if "cri.maintainer.ext" not in have:
        ext.write_text("\n".join(have + ["cri.maintainer.ext"]) + "\n")
    print("Saved to", push.CONFIG, "(chmod 600). To switch it off: delete that file.")


if __name__ == "__main__":
    main()
