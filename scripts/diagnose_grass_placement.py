from __future__ import annotations

from maya import cmds


def find_mash_networks():
    """find all MASH waiter nodes in the scene."""
    waiters = cmds.ls(type="MASH_Waiter") or []
    return waiters


def get_transform_info(node):
    """get world-space transform info for a node."""
    # walk up to the transform parent if needed
    if cmds.nodeType(node) != "transform":
        parents = cmds.listRelatives(node, parent=True, fullPath=True) or []
        if parents:
            node = parents[0]
        else:
            return None

    try:
        translate = cmds.xform(node, q=True, worldSpace=True, translation=True)
        pivot = cmds.xform(node, q=True, worldSpace=True, rotatePivot=True)
        scale = cmds.xform(node, q=True, worldSpace=True, scale=True)
        return {
            "node": node,
            "translate": translate,
            "pivot": pivot,
            "scale": scale,
        }
    except Exception as e:
        return {"node": node, "error": str(e)}


def get_mash_python_positions(python_node):
    """extract the embedded position list from a MASH python node's pyScript."""
    try:
        code = cmds.getAttr(f"{python_node}.pyScript")
        if not code:
            return None

        # find the positions = [...] line and eval it safely
        import ast
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("positions = ["):
                # might span multiple lines — grab from 'positions = ' to end
                start = code.index("positions = [")
                # find matching bracket
                bracket_count = 0
                end = start
                for i, ch in enumerate(code[start:], start):
                    if ch == "[":
                        bracket_count += 1
                    elif ch == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end = i + 1
                            break
                expr = code[start:end].split("=", 1)[1].strip()
                positions = ast.literal_eval(expr)
                return positions
    except Exception as e:
        return f"error parsing: {e}"
    return None


def diagnose():
    """run full diagnostic. prints report to script editor."""
    print("\n" + "=" * 70)
    print("  GRASS PLACEMENT DIAGNOSTIC")
    print("=" * 70)

    # 1. find terrain meshes (heuristic: largest mesh by bbox volume)
    all_meshes = cmds.ls(type="mesh") or []
    mesh_transforms = []
    for m in all_meshes:
        parents = cmds.listRelatives(m, parent=True, fullPath=False) or []
        if parents:
            mesh_transforms.append(parents[0])
    mesh_transforms = list(set(mesh_transforms))

    print(f"\n[scene] {len(mesh_transforms)} mesh transforms found")

    # 2. find MASH networks
    waiters = find_mash_networks()
    print(f"[scene] {len(waiters)} MASH networks found: {waiters}")

    if not waiters:
        print("\n  !! no MASH networks in scene. generate grass first.")
        return

    for waiter in waiters:
        print(f"\n{'─' * 60}")
        print(f"  MASH NETWORK: {waiter}")
        print(f"{'─' * 60}")

        # waiter transform
        info = get_transform_info(waiter)
        if info and "error" not in info:
            t = info["translate"]
            p = info["pivot"]
            print(f"\n  [waiter transform]")
            print(f"    translate (world): ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
            print(f"    pivot (world):     ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
            offset_mag = (t[0]**2 + t[1]**2 + t[2]**2) ** 0.5
            if offset_mag > 0.01:
                print(f"    !! WAITER IS NOT AT ORIGIN (offset={offset_mag:.2f})")
                print(f"    !! this will shift ALL grass instances by ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")

        # find connected instancer / repro mesh
        instancer_nodes = cmds.ls(f"{waiter}*", type="MASH_Repro") or []
        if not instancer_nodes:
            # try broader search
            instancer_nodes = cmds.ls(f"{waiter.split('_')[0]}*Repro*") or []
        for inst in instancer_nodes:
            info = get_transform_info(inst)
            if info and "error" not in info:
                t = info["translate"]
                print(f"\n  [repro node: {inst}]")
                print(f"    translate (world): ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
                offset_mag = (t[0]**2 + t[1]**2 + t[2]**2) ** 0.5
                if offset_mag > 0.01:
                    print(f"    !! REPRO IS NOT AT ORIGIN (offset={offset_mag:.2f})")

        # find the instanced geometry (what's being scattered)
        repro_shapes = cmds.ls(f"{waiter}*", type="MASH_Repro") or []
        for repro in repro_shapes:
            connections = cmds.listConnections(f"{repro}.instancedGroup", source=True) or []
            for conn in connections:
                info = get_transform_info(conn)
                if info and "error" not in info:
                    t = info["translate"]
                    print(f"\n  [instanced geometry: {conn}]")
                    print(f"    translate (world): ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
                    offset_mag = (t[0]**2 + t[1]**2 + t[2]**2) ** 0.5
                    if offset_mag > 0.01:
                        print(f"    !! SOURCE GEO IS NOT AT ORIGIN (offset={offset_mag:.2f})")
                        print(f"    !! this may shift all instances")

        # find MASH_Python node for this network
        python_nodes = cmds.ls(f"{waiter}*", type="MASH_Python") or []
        if not python_nodes:
            python_nodes = cmds.ls(type="MASH_Python") or []
            python_nodes = [p for p in python_nodes if waiter.replace("_Waiter", "") in p]

        for pynode in python_nodes:
            print(f"\n  [python node: {pynode}]")
            positions = get_mash_python_positions(pynode)
            if isinstance(positions, str):
                print(f"    {positions}")
            elif positions and len(positions) > 0:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                zs = [p[2] for p in positions]
                print(f"    embedded point count: {len(positions)}")
                print(f"    x range: [{min(xs):.2f}, {max(xs):.2f}]")
                print(f"    y range: [{min(ys):.2f}, {max(ys):.2f}]")
                print(f"    z range: [{min(zs):.2f}, {max(zs):.2f}]")
                print(f"    centroid: ({sum(xs)/len(xs):.2f}, {sum(ys)/len(ys):.2f}, {sum(zs)/len(zs):.2f})")
            else:
                print(f"    no positions found in pyScript")

        # find the distribute node
        dist_nodes = cmds.ls(f"{waiter}*", type="MASH_Distribute") or []
        for dist in dist_nodes:
            print(f"\n  [distribute node: {dist}]")
            try:
                pc = cmds.getAttr(f"{dist}.pointCount")
                print(f"    point count: {pc}")
            except Exception:
                pass

    # 3. check all mesh transforms that might be terrain
    print(f"\n{'─' * 60}")
    print(f"  MESH TRANSFORMS (potential terrain)")
    print(f"{'─' * 60}")

    for mesh_t in sorted(mesh_transforms):
        # skip MASH internal meshes
        if "MASH" in mesh_t or "Repro" in mesh_t:
            continue
        try:
            bbox = cmds.exactWorldBoundingBox(mesh_t)
            t = cmds.xform(mesh_t, q=True, worldSpace=True, translation=True)
            p = cmds.xform(mesh_t, q=True, worldSpace=True, rotatePivot=True)

            bbox_width = bbox[3] - bbox[0]
            bbox_depth = bbox[5] - bbox[2]
            bbox_height = bbox[4] - bbox[1]

            print(f"\n  [{mesh_t}]")
            print(f"    translate (world): ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})")
            print(f"    pivot (world):     ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
            print(f"    bbox:  x=[{bbox[0]:.2f}, {bbox[3]:.2f}] y=[{bbox[1]:.2f}, {bbox[4]:.2f}] z=[{bbox[2]:.2f}, {bbox[5]:.2f}]")
            print(f"    size:  {bbox_width:.2f} x {bbox_height:.2f} x {bbox_depth:.2f}")

            # check if pivot matches bbox center (frozen transforms)
            bbox_center_x = (bbox[0] + bbox[3]) / 2
            bbox_center_z = (bbox[2] + bbox[5]) / 2
            pivot_offset_x = p[0] - bbox_center_x
            pivot_offset_z = p[2] - bbox_center_z
            if abs(pivot_offset_x) > 1.0 or abs(pivot_offset_z) > 1.0:
                print(f"    note: pivot is offset from bbox center by ({pivot_offset_x:.2f}, {pivot_offset_z:.2f})")
        except Exception as e:
            print(f"\n  [{mesh_t}] error: {e}")

    print(f"\n{'=' * 70}")
    print("  END DIAGNOSTIC")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    diagnose()
