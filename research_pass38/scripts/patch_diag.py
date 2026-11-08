import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
a = p.parse_args()
path = a.source / "av1/encoder/tx_search.c"
s = path.read_text(encoding="utf-8")

def replace_once(old, new, label):
    global s
    if s.count(old) != 1:
        raise SystemExit(f"{label} anchor mismatch: {s.count(old)}")
    s = s.replace(old, new, 1)

replace_once(
    '#include "av1/encoder/txb_rdopt.h"\n',
    '#include "av1/encoder/txb_rdopt.h"\n\n#ifdef PBC38_DIAG\n#include <stdio.h>\n#include <stdlib.h>\n#endif\n',
    "include",
)
replace_once(
    "#define PROB_THRESH_OFFSET_TX_TYPE 100\n",
    """#define PROB_THRESH_OFFSET_TX_TYPE 100

#ifdef PBC38_DIAG
static unsigned int pbc38_eligible_searches;
static unsigned int pbc38_pruned_searches;
static unsigned int pbc38_hash_selected;
static unsigned int pbc38_sampled_total;
static unsigned int pbc38_better_total;
static unsigned int pbc38_complete_total;
static unsigned int pbc38_attempt_total;
static unsigned int pbc38_full_rd_total;

static void pbc38_report(void) {
  fprintf(stderr, "PBC38_SUMMARY,%u,%u,%u,%u,%u,%u,%u,%u\\n",
          pbc38_eligible_searches, pbc38_pruned_searches,
          pbc38_hash_selected, pbc38_sampled_total, pbc38_better_total,
          pbc38_complete_total, pbc38_attempt_total, pbc38_full_rd_total);
}
#endif
""",
    "summary",
)
replace_once(
    "  int rate_cost = 0;\n  struct macroblock_plane *const p = &x->plane[plane];",
    """  int rate_cost = 0;
#ifdef PBC38_DIAG
  static unsigned int pbc38_sampled_events = 0;
  static int pbc38_registered = 0;
  int pbc38_shadow_active = 0;
  int pbc38_shadow_complete = 1;
  int pbc38_shadow_attempts = 0;
  int pbc38_shadow_full_rd = 0;
  int64_t pbc38_cutoff_rd = INT64_MAX;
  int64_t pbc38_shadow_best_rd = INT64_MAX;
  int64_t pbc38_shadow_best_dist = 0;
  int pbc38_shadow_best_rate = 0;
  TX_TYPE pbc38_cutoff_type = DCT_DCT;
  TX_TYPE pbc38_shadow_best_type = DCT_DCT;
  unsigned int pbc38_sample_id = 0;
#endif
  struct macroblock_plane *const p = &x->plane[plane];""",
    "locals",
)
replace_once(
    "  const uint16_t allowed_tx_mask = tx_mask;\n",
    """  const uint16_t allowed_tx_mask = tx_mask;
#ifdef PBC38_DIAG
  unsigned int pbc38_key = (unsigned int)xd->mi_row * 2654435761u;
  pbc38_key ^= (unsigned int)xd->mi_col * 2246822519u;
  pbc38_key ^= (unsigned int)tx_size * 3266489917u;
  pbc38_key ^= (unsigned int)blk_row * 668265263u;
  pbc38_key ^= (unsigned int)blk_col * 374761393u;
  pbc38_key ^= (unsigned int)mbmi->mode * 1274126177u;
  pbc38_key ^= (unsigned int)plane * 42595009u;
  pbc38_key ^= pbc38_key >> 16;
  const int pbc38_eligible =
      plane == 0 && !is_inter_block(mbmi) &&
      cpi->sf.tx_sf.adaptive_txb_search_level > 0 &&
      __builtin_popcount((unsigned int)allowed_tx_mask) > 1;
  if (pbc38_eligible) ++pbc38_eligible_searches;
  if (!pbc38_registered) {
    atexit(pbc38_report);
    pbc38_registered = 1;
  }
#endif
""",
    "eligibility",
)
replace_once(
    "    if (tx_type == TX_TYPE_INVALID || !check_bit_mask(allowed_tx_mask, tx_type))\n      continue;\n    txfm_param.tx_type = tx_type;",
    """    if (tx_type == TX_TYPE_INVALID || !check_bit_mask(allowed_tx_mask, tx_type))
      continue;
#ifdef PBC38_DIAG
    if (pbc38_shadow_active) ++pbc38_shadow_attempts;
#endif
    txfm_param.tx_type = tx_type;""",
    "candidate",
)
replace_once(
    "    // Calculate distortion.\n    if (eobs_ptr[block] == 0) {",
    """#ifdef PBC38_DIAG
    if (pbc38_shadow_active) ++pbc38_shadow_full_rd;
#endif
    // Calculate distortion.
    if (eobs_ptr[block] == 0) {""",
    "distortion",
)
replace_once(
    "    if (rd < best_rd) {\n      best_rd = rd;",
    """#ifdef PBC38_DIAG
    if (pbc38_shadow_active) {
      if (rd < pbc38_shadow_best_rd) {
        pbc38_shadow_best_rd = rd;
        pbc38_shadow_best_type = tx_type;
        pbc38_shadow_best_rate = this_rd_stats.rate;
        pbc38_shadow_best_dist = this_rd_stats.dist;
      }
    } else
#endif
    if (rd < best_rd) {
      best_rd = rd;""",
    "winner",
)
replace_once(
    """    if (cpi->sf.tx_sf.adaptive_txb_search_level) {
      if ((best_rd - (best_rd >> cpi->sf.tx_sf.adaptive_txb_search_level)) >
          ref_best_rd) {
        break;
      }
    }

    // Terminate transform type search if the block has been quantized to
    // all zero.
    if (cpi->sf.tx_sf.tx_type_search.skip_tx_search && !best_eob) break;""",
    """    if (!pbc38_shadow_active && cpi->sf.tx_sf.adaptive_txb_search_level) {
      if ((best_rd - (best_rd >> cpi->sf.tx_sf.adaptive_txb_search_level)) >
          ref_best_rd) {
#ifdef PBC38_DIAG
        if (pbc38_eligible) ++pbc38_pruned_searches;
        if (pbc38_eligible && pbc38_key % 16u == 0) {
          ++pbc38_hash_selected;
        }
        if (pbc38_eligible && pbc38_key % 16u == 0 &&
            pbc38_sampled_events < 128) {
          pbc38_sample_id = ++pbc38_sampled_events;
          ++pbc38_sampled_total;
          pbc38_shadow_active = 1;
          pbc38_cutoff_rd = best_rd;
          pbc38_shadow_best_rd = best_rd;
          pbc38_cutoff_type = best_tx_type;
          pbc38_shadow_best_type = best_tx_type;
          pbc38_shadow_best_rate = best_rd_stats->rate;
          pbc38_shadow_best_dist = best_rd_stats->dist;
        } else {
          break;
        }
#else
        break;
#endif
      }
    }

    // Terminate transform type search if the block has been quantized to
    // all zero.
    if (cpi->sf.tx_sf.tx_type_search.skip_tx_search && !best_eob) {
#ifdef PBC38_DIAG
      if (pbc38_shadow_active) pbc38_shadow_complete = 0;
#endif
      break;
    }""",
    "adaptive",
)
replace_once(
    "  assert(best_rd != INT64_MAX);\n\n  best_rd_stats->skip_txfm = best_eob == 0;",
    """#ifdef PBC38_DIAG
  if (pbc38_shadow_active) {
    pbc38_attempt_total += (unsigned int)pbc38_shadow_attempts;
    pbc38_full_rd_total += pbc38_shadow_full_rd;
    if (pbc38_shadow_complete) ++pbc38_complete_total;
    if (pbc38_shadow_best_rd < pbc38_cutoff_rd) ++pbc38_better_total;
    fprintf(stderr,
            "PBC38_EVENT,%u,%d,%d,%d,%d,%d,%d,%d,%" PRId64
            ",%d,%d,%" PRId64 ",%" PRId64 ",%d,%d,%" PRId64
            ",%" PRId64 ",%d,%d,%d\\n",
            pbc38_sample_id, xd->mi_row, xd->mi_col, (int)tx_size, blk_row,
            blk_col, (int)mbmi->mode, x->rdmult, pbc38_cutoff_rd,
            (int)pbc38_cutoff_type, best_rd_stats->rate, best_rd_stats->dist,
            pbc38_shadow_best_rd, (int)pbc38_shadow_best_type,
            pbc38_shadow_best_rate, pbc38_shadow_best_dist,
            pbc38_shadow_best_rd - pbc38_cutoff_rd,
            pbc38_shadow_attempts, pbc38_shadow_full_rd,
            pbc38_shadow_complete);
  }
#endif
  assert(best_rd != INT64_MAX);

  best_rd_stats->skip_txfm = best_eob == 0;""",
    "output",
)
path.write_text(s, encoding="utf-8", newline="")
