from pathlib import Path
import argparse
p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
a = p.parse_args()
f = a.source / "av1/encoder/tx_search.c"
s = f.read_text(encoding="utf-8")
old = """    if (cpi->sf.tx_sf.adaptive_txb_search_level) {
      if ((best_rd - (best_rd >> cpi->sf.tx_sf.adaptive_txb_search_level)) >
          ref_best_rd) {
        break;
      }
    }"""
new = """#if !defined(PBC38_EXPAND)
    if (cpi->sf.tx_sf.adaptive_txb_search_level) {
      if ((best_rd - (best_rd >> cpi->sf.tx_sf.adaptive_txb_search_level)) >
          ref_best_rd) {
        break;
      }
    }
#endif"""
if s.count(old) != 1:
    raise SystemExit(f"adaptive-exit anchor mismatch: {s.count(old)}")
f.write_text(s.replace(old, new, 1), encoding="utf-8", newline="")
