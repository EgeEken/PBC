import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[2]
out = root / "research_pass44" / "outputs"
joint = json.loads((out / "joint_allocation_search.json").read_text())
full = json.loads((out / "full_prefix_budget_search.json").read_text())
screen = json.loads((out / "strict_budget_refinement.json").read_text())
full_by_key = {(r["image"], r["point"]): r for r in full["results"]}
base_by_key = {}
for r in screen["rows"]:
    for x in r["tested"]:
        if x["selected_regions"] == 0:
            base_by_key[(r["image"], r["point"], x["coarse_patch_count"])] = x
rows = []
for r in joint["results"]:
    image, point = r["image"], r["point"]
    choices = [dict(c, base_only=base_by_key[(image, point, c["coarse_patch_count"])]) for c in r["coarse_choices"] if c["coarse_patch_count"] != 13]
    if image.startswith("googlewiki"):
        full_choice = dict(next(c for c in r["coarse_choices"] if c["coarse_patch_count"] == 12), base_only=base_by_key[(image, point, 12)])
        choices.append(full_choice)
    else:
        f = full_by_key[("cat.jpg" if image.startswith("cat") else "coast.jpg", point)]
        choices.append({"coarse_patch_count": f["coarse_patch_count"], "base_only": f["base_only"], "best_feasible": f["best_feasible"], "search_evaluations": f["search_evaluations"]})
    choices.sort(key=lambda c: c["coarse_patch_count"])
    a = min((c["base_only"] for c in choices), key=lambda x: x["rgb_mse"])
    base_choice = next(c for c in choices if c["base_only"]["file_sha256"] == a["file_sha256"])
    b = base_choice["best_feasible"]
    c = min((choice["best_feasible"] for choice in choices), key=lambda x: x["rgb_mse"])
    row = {"image": image, "point": point, "target_preset": r["target_preset"], "target_bytes": r["target_bytes"], "hard_cap_bytes": r["hard_cap_bytes"], "A_best_coarse_only": a, "B_fixed_best_base_plus_detail": b, "B_coarse_patch_count": base_choice["coarse_patch_count"], "C_joint_best": c, "C_coarse_patch_count": next(ch["coarse_patch_count"] for ch in choices if ch["best_feasible"]["file_sha256"] == c["file_sha256"]), "choices": choices}
    rows.append(row)
result = {"method": "Corrected joint choice menu: Googlewiki p3/p6/full p12; Cat p3/p6/full p19; Coast p3/p6/full p50. Candidate selections use actual completed-file bytes and RGB MSE. Cat/Coast full-prefix subset search was bounded to K<=512 and ten completed files per target.", "hard_budget": "105% of matching frozen V3.1 preset file bytes", "results": rows}
(out / "final_joint_allocation.json").write_text(json.dumps(result, indent=2) + "\n")
for r in rows:
    def fmt(x): return f'{x["bytes"]} / {x["rgb_mse"]:.3f}'
    print(r["image"], r["point"], "A", fmt(r["A_best_coarse_only"]), "p", r["C_coarse_patch_count"], "B", fmt(r["B_fixed_best_base_plus_detail"]), "C", fmt(r["C_joint_best"]), "p", r["C_coarse_patch_count"])
