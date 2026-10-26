import shutil

from support import OUT, save, snapshot

destination=OUT/'invalid_runtime_label'
destination.mkdir(exist_ok=False)
paths=[p for p in OUT.iterdir() if p.is_file() and
    (p.name.startswith('matched_confirmation') or p.name.startswith('efficiency_profile') or p.name.startswith('allocator_profile_functions'))]
for path in paths:
    shutil.copy2(path,destination/path.name)
save('routing_correction',dict(snapshot=snapshot(),archived=[str(p.relative_to(OUT)) for p in paths],
    error="startswith('all') also matched 'alloc', enabling grid/search in allocator timing controls",
    affected='First matched_confirmation timing and first efficiency_profile; data retained in invalid_runtime_label; excluded from final numerical conclusions',
    unaffected='A four modes, fresh ordinary controls, frozen Pass3 allocator bitstreams/RD and custom regression diagnostics. Runtime bit-exactness makes first allocation outputs identical but does not make unequal-path timings valid.',
    correction="all_runtime = name == 'all' or name.startswith('all_'); rerun matched_confirmation and efficiency_profile with identical generic runtime flags for ordinary/allocator."))
