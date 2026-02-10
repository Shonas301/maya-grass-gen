"""diagnostic script for grass generation.

paste this into maya's script editor (python tab) and run it.
reads the same UI preferences so it uses your current settings.
prints detailed diagnostics at each pipeline stage.
"""

from maya import cmds


def _pref(name, default):
    """read a grass UI preference."""
    key = f"grassUI_{name}"
    if cmds.optionVar(exists=key):
        return cmds.optionVar(query=key)
    return default


def diagnose():
    """run grass generation with full diagnostics."""
    terrain = _pref("terrain", "")
    grass = _pref("grass", "")

    if not terrain:
        print("[DIAG] ERROR: no terrain mesh set in UI prefs. "
              "open the grass UI, set a terrain, and try again.")
        return
    if not grass:
        print("[DIAG] ERROR: no grass geometry set in UI prefs. "
              "open the grass UI, set grass geo, and try again.")
        return

    print("=" * 60)
    print("[DIAG] grass generation diagnostics")
    print("=" * 60)

    # check meshes exist
    for label, name in [("terrain", terrain), ("grass geo", grass)]:
        if not cmds.objExists(name):
            print(f"[DIAG] ERROR: {label} '{name}' not found in scene!")
            return
        shapes = cmds.listRelatives(name, shapes=True, type="mesh") or []
        faces = cmds.polyEvaluate(name, face=True) if shapes else 0
        bbox = cmds.exactWorldBoundingBox(name)
        print(f"[DIAG] {label}: '{name}' — {len(shapes)} shape(s), "
              f"{faces} faces, bbox=({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f}) "
              f"to ({bbox[3]:.1f},{bbox[4]:.1f},{bbox[5]:.1f})")

    # read params from UI prefs
    params = {
        "terrain_mesh": terrain,
        "grass_geometry": grass,
        "count": int(_pref("count", 5000)),
        "wind_strength": float(_pref("wind", 2.5)),
        "scale_variation_wave1": (
            float(_pref("scale_min_w1", 0.8)),
            float(_pref("scale_max_w1", 1.2)),
        ),
        "scale_variation_wave2": (
            float(_pref("scale_min_w2", 0.8)),
            float(_pref("scale_max_w2", 1.2)),
        ),
        "seed": int(_pref("seed", 42)),
        "noise_scale": float(_pref("noise", 0.004)),
        "time_scale": float(_pref("time", 0.008)),
        "octaves": int(_pref("octaves", 4)),
        "persistence": float(_pref("persistence", 0.5)),
        "proximity_density_boost": float(_pref("proximity", 1.0)),
        "min_distance": float(_pref("min_distance", 5.0)),
        "max_lean_angle": float(_pref("max_lean", 30.0)),
        "cluster_falloff": float(_pref("cluster_falloff", 0.5)),
        "edge_offset": float(_pref("edge_offset", 10.0)),
        "gravity_weight": float(_pref("gravity_weight", 0.75)),
        "verbose": True,
    }

    print()
    print("[DIAG] parameters:")
    for k, v in params.items():
        if k != "verbose":
            print(f"  {k}: {v}")

    print()
    print("-" * 60)
    print("[DIAG] running generate_grass with verbose=True ...")
    print("-" * 60)

    try:
        from maya_grass_gen import generate_grass
        network_name = generate_grass(**params)
    except Exception as exc:
        print(f"\n[DIAG] FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("-" * 60)
    print("[DIAG] post-generation checks")
    print("-" * 60)

    # check MASH network exists
    if not cmds.objExists(network_name):
        print(f"[DIAG] WARNING: network '{network_name}' not found after creation")
    else:
        print(f"[DIAG] network '{network_name}' exists")

    # find the MASH waiter (instancer) node
    waiters = cmds.ls(f"{network_name}*", type="MASH_Waiter") or []
    print(f"[DIAG] MASH_Waiter nodes: {waiters}")
    for w in waiters:
        point_count = cmds.getAttr(f"{w}.pointCount")
        print(f"[DIAG]   {w}.pointCount = {point_count}")

    # find MASH python nodes and check for errors
    pythons = cmds.ls(f"{network_name}*", type="MASH_Python") or []
    if not pythons:
        # broader search
        pythons = cmds.ls(type="MASH_Python") or []
    print(f"[DIAG] MASH_Python nodes: {pythons}")
    for p in pythons:
        script = cmds.getAttr(f"{p}.pyScript") or ""
        lines = script.strip().split("\n")
        print(f"[DIAG]   {p}: {len(lines)} lines of code")
        # check if positions list is empty
        for line in lines:
            if line.startswith("positions = "):
                # count items
                if line.strip() == "positions = []":
                    print(f"[DIAG]   WARNING: positions list is EMPTY — no grass points!")
                else:
                    # rough count of tuples
                    count = line.count("(")
                    print(f"[DIAG]   positions: ~{count} points baked in")
                break

    # check the Repro mesh (MASH instancer output)
    repros = cmds.ls(f"{network_name}*Repro*", type="mesh") or []
    if not repros:
        repros = cmds.ls(f"*Repro*", type="mesh") or []
    print(f"[DIAG] Repro meshes: {repros}")
    for r in repros:
        parent = (cmds.listRelatives(r, parent=True) or [None])[0]
        vis = cmds.getAttr(f"{parent}.visibility") if parent else "?"
        print(f"[DIAG]   {r} — parent={parent}, visible={vis}")

    # check if grass geo was re-centered
    bbox = cmds.exactWorldBoundingBox(grass)
    cx = (bbox[0] + bbox[3]) / 2.0
    cy = (bbox[1] + bbox[4]) / 2.0
    cz = (bbox[2] + bbox[5]) / 2.0
    print(f"[DIAG] grass geo '{grass}' bbox center after generation: "
          f"({cx:.3f}, {cy:.3f}, {cz:.3f})")
    if abs(cx) > 1 or abs(cy) > 1 or abs(cz) > 1:
        print(f"[DIAG]   WARNING: grass geo center is far from origin — "
              f"MASH instances may appear offset")

    print()
    print("=" * 60)
    print("[DIAG] diagnostics complete — paste output above to share")
    print("=" * 60)


diagnose()
