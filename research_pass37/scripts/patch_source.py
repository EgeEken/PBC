import pathlib
import sys

source = pathlib.Path(sys.argv[1]) / "av1/encoder/intra_mode_search.c"
text = source.read_text(encoding="utf-8")
old = """  for (int mode_idx = INTRA_MODE_START; mode_idx < LUMA_MODE_COUNT;
       ++mode_idx) {
    set_y_mode_and_delta_angle(mode_idx, mbmi,
                               intra_sf->prune_luma_odd_delta_angles_in_intra);"""
new = """#ifdef PBC37_EXPERIMENT
  int intra_mode_order[INTRA_MODE_NUM];
  for (int i = 0; i < INTRA_MODE_NUM; ++i) intra_mode_order[i] = i;
  for (int i = 1; i < INTRA_MODE_NUM; ++i) {
    const int key = intra_mode_order[i];
    const int key_cost = bmode_costs[intra_rd_search_mode_order[key]];
    int j = i - 1;
    while (j >= 0 &&
           bmode_costs[intra_rd_search_mode_order[intra_mode_order[j]]] >
               key_cost) {
      intra_mode_order[j + 1] = intra_mode_order[j];
      --j;
    }
    intra_mode_order[j + 1] = key;
  }
#endif
  for (int mode_idx = INTRA_MODE_START; mode_idx < LUMA_MODE_COUNT;
       ++mode_idx) {
#ifdef PBC37_EXPERIMENT
    const int candidate_idx =
        mode_idx < INTRA_MODE_END
            ? intra_mode_order[mode_idx - INTRA_MODE_START]
            : mode_idx;
#else
    const int candidate_idx = mode_idx;
#endif
    set_y_mode_and_delta_angle(candidate_idx, mbmi,
                               intra_sf->prune_luma_odd_delta_angles_in_intra);"""
if text.count(old) != 1:
    raise SystemExit(f"expected one loop anchor, found {text.count(old)}")
source.write_text(text.replace(old, new), encoding="utf-8", newline="")
