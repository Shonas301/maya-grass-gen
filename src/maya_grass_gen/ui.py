"""Maya UI for grass generator.

Provides a graphical interface for the grass generation system, allowing users
to configure parameters and generate grass without writing Python code.

Usage:
    from maya_grass_gen import show_grass_ui
    show_grass_ui()
"""

import functools
from maya import cmds


# window constants
WINDOW_NAME = "grassGeneratorUI"
PREF_PREFIX = "grassUI_"


def load_pref(pref_name: str, default):
    """Load a preference value from Maya's optionVar system.

    Args:
        pref_name: name of the preference (will be prefixed with grassUI_)
        default: default value if preference doesn't exist

    Returns:
        stored value or default if not set
    """
    key = f"{PREF_PREFIX}{pref_name}"
    if not cmds.optionVar(exists=key):
        return default

    # determine type from default and query accordingly
    if isinstance(default, int):
        return cmds.optionVar(query=key)
    elif isinstance(default, float):
        return cmds.optionVar(query=key)
    elif isinstance(default, str):
        return cmds.optionVar(query=key)
    else:
        return default


def save_pref(pref_name: str, value) -> None:
    """Save a preference value to Maya's optionVar system.

    Args:
        pref_name: name of the preference (will be prefixed with grassUI_)
        value: value to store (int, float, or str)
    """
    key = f"{PREF_PREFIX}{pref_name}"

    if isinstance(value, int):
        cmds.optionVar(intValue=(key, value))
    elif isinstance(value, float):
        cmds.optionVar(floatValue=(key, value))
    elif isinstance(value, str):
        cmds.optionVar(stringValue=(key, value))


def set_selected_mesh(field_name: str, *args) -> None:
    """Populate a text field with the currently selected mesh.

    Args:
        field_name: name of the textFieldButtonGrp control to update
        *args: swallow Maya's default callback arguments
    """
    # get current selection
    selection = cmds.ls(selection=True, type="transform")

    if not selection:
        cmds.warning("No object selected. Please select a mesh.")
        return

    mesh_name = selection[0]

    # verify it has a mesh shape
    shapes = cmds.listRelatives(mesh_name, shapes=True, type="mesh")
    if not shapes:
        cmds.warning(f"'{mesh_name}' is not a mesh. Please select a polygon mesh.")
        return

    # update the text field
    cmds.textFieldButtonGrp(field_name, edit=True, text=mesh_name)


def execute_grass_generation(*args) -> None:
    """Execute grass generation with current UI values.

    Args:
        *args: swallow Maya's default callback arguments
    """
    # read UI values
    terrain = cmds.textFieldButtonGrp("terrain_field", query=True, text=True)
    grass = cmds.textFieldButtonGrp("grass_field", query=True, text=True)

    # validate required fields
    if not terrain or not terrain.strip():
        cmds.error("Please specify a terrain mesh.")
        return
    if not grass or not grass.strip():
        cmds.error("Please specify grass geometry.")
        return

    # read parameter values
    count = cmds.intSliderGrp("count_field", query=True, value=True)
    wind = cmds.floatSliderGrp("wind_field", query=True, value=True)
    scale_min_w1 = cmds.floatSliderGrp("scale_min_w1", query=True, value=True)
    scale_max_w1 = cmds.floatSliderGrp("scale_max_w1", query=True, value=True)
    scale_min_w2 = cmds.floatSliderGrp("scale_min_w2", query=True, value=True)
    scale_max_w2 = cmds.floatSliderGrp("scale_max_w2", query=True, value=True)
    proximity = cmds.floatSliderGrp("proximity_field", query=True, value=True)
    seed = cmds.intFieldGrp("seed_field", query=True, value1=True)
    noise = cmds.floatFieldGrp("noise_field", query=True, value1=True)
    time_scale = cmds.floatFieldGrp("time_field", query=True, value1=True)
    octaves = cmds.intSliderGrp("octaves_field", query=True, value=True)
    persistence = cmds.floatSliderGrp("persistence_field", query=True, value=True)
    max_lean = cmds.floatSliderGrp("max_lean_field", query=True, value=True)
    min_distance = cmds.floatSliderGrp("min_distance_field", query=True, value=True)
    cluster_falloff = cmds.floatSliderGrp("cluster_falloff_field", query=True, value=True)
    edge_offset = cmds.floatSliderGrp("edge_offset_field", query=True, value=True)
    gravity_weight = cmds.floatSliderGrp("gravity_weight_field", query=True, value=True)

    # save all preferences
    save_pref("terrain", terrain)
    save_pref("grass", grass)
    save_pref("count", count)
    save_pref("wind", wind)
    save_pref("scale_min_w1", scale_min_w1)
    save_pref("scale_max_w1", scale_max_w1)
    save_pref("scale_min_w2", scale_min_w2)
    save_pref("scale_max_w2", scale_max_w2)
    save_pref("proximity", proximity)
    save_pref("seed", seed)
    save_pref("noise", noise)
    save_pref("time", time_scale)
    save_pref("octaves", octaves)
    save_pref("persistence", persistence)
    save_pref("max_lean", max_lean)
    save_pref("min_distance", min_distance)
    save_pref("cluster_falloff", cluster_falloff)
    save_pref("edge_offset", edge_offset)
    save_pref("gravity_weight", gravity_weight)

    # execute generation
    try:
        from maya_grass_gen import generate_grass

        network_name = generate_grass(
            terrain_mesh=terrain,
            grass_geometry=grass,
            count=count,
            wind_strength=wind,
            scale_variation_wave1=(scale_min_w1, scale_max_w1),
            scale_variation_wave2=(scale_min_w2, scale_max_w2),
            seed=seed,
            noise_scale=noise,
            time_scale=time_scale,
            proximity_density_boost=proximity,
            octaves=octaves,
            min_distance=min_distance,
            max_lean_angle=max_lean,
            cluster_falloff=cluster_falloff,
            edge_offset=edge_offset,
            persistence=persistence,
            gravity_weight=gravity_weight,
        )

        # show success message
        message = f"Grass generation complete!\n\nMASH network created: {network_name}"
        cmds.confirmDialog(
            title="Success",
            message=message,
            button=["OK"],
            defaultButton="OK",
        )

    except Exception as e:
        # show error message
        cmds.confirmDialog(
            title="Error",
            message=f"Grass generation failed:\n\n{str(e)}",
            button=["OK"],
            defaultButton="OK",
            icon="critical",
        )
        raise


def show_grass_ui() -> None:
    """Show the grass generator UI window.

    Creates a singleton window with controls for all grass generation parameters.
    Parameter values are persisted between Maya sessions using optionVars.
    """
    # singleton pattern - delete existing window
    if cmds.window(WINDOW_NAME, exists=True):
        cmds.deleteUI(WINDOW_NAME)

    # create main window
    window = cmds.window(
        WINDOW_NAME,
        title="Grass Generator",
        widthHeight=(420, 780),
        sizeable=False,
    )

    # main layout
    cmds.columnLayout(adjustableColumn=True, rowSpacing=5)

    # scene objects section
    cmds.frameLayout(label="Scene Objects", collapsable=True, collapse=False)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=3)

    cmds.textFieldButtonGrp(
        "terrain_field",
        label="Terrain Mesh:",
        text=load_pref("terrain", ""),
        buttonLabel="Set Selected",
        buttonCommand=functools.partial(set_selected_mesh, "terrain_field"),
        columnWidth3=(100, 200, 100),
        annotation="The ground mesh to scatter grass onto. Select a mesh and click 'Set Selected'.",
    )

    cmds.textFieldButtonGrp(
        "grass_field",
        label="Grass Geometry:",
        text=load_pref("grass", ""),
        buttonLabel="Set Selected",
        buttonCommand=functools.partial(set_selected_mesh, "grass_field"),
        columnWidth3=(100, 200, 100),
        annotation="The grass blade mesh to instance. Select your grass geometry and click 'Set Selected'.",
    )

    cmds.setParent('..')  # exit columnLayout
    cmds.setParent('..')  # exit frameLayout

    # grass parameters section
    cmds.frameLayout(label="Grass Parameters", collapsable=True, collapse=False)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=3)

    cmds.intSliderGrp(
        "count_field",
        label="Blade Count:",
        field=True,
        minValue=100,
        maxValue=20000,
        fieldMinValue=1,
        fieldMaxValue=100000,
        value=load_pref("count", 5000),
        columnWidth3=(100, 50, 250),
        annotation="Total number of grass blades to generate. Higher values give denser coverage but take longer.",
    )

    cmds.floatSliderGrp(
        "wind_field",
        label="Wind Strength:",
        field=True,
        minValue=0.0,
        maxValue=10.0,
        fieldMinValue=0.0,
        fieldMaxValue=50.0,
        value=load_pref("wind", 2.5),
        precision=1,
        columnWidth3=(100, 50, 250),
        annotation="How strongly the wind pushes grass blades. 0 = no wind, 2.5 = moderate breeze, 10 = strong gusts.",
    )

    cmds.floatSliderGrp(
        "scale_min_w1",
        label="Scale Min (Wave 1):",
        field=True,
        minValue=0.1,
        maxValue=20.0,
        value=load_pref("scale_min_w1", 0.8),
        precision=2,
        columnWidth3=(130, 50, 220),
        annotation="Minimum scale for grass in open areas (away from obstacles). Lower = smaller blades.",
    )

    cmds.floatSliderGrp(
        "scale_max_w1",
        label="Scale Max (Wave 1):",
        field=True,
        minValue=0.1,
        maxValue=20.0,
        value=load_pref("scale_max_w1", 1.2),
        precision=2,
        columnWidth3=(130, 50, 220),
        annotation="Maximum scale for grass in open areas (away from obstacles). Higher = larger blades.",
    )

    cmds.floatSliderGrp(
        "scale_min_w2",
        label="Scale Min (Wave 2):",
        field=True,
        minValue=0.1,
        maxValue=20.0,
        value=load_pref("scale_min_w2", 0.8),
        precision=2,
        columnWidth3=(130, 50, 220),
        annotation="Minimum scale for grass near obstacles. Use smaller values for shorter grass around objects.",
    )

    cmds.floatSliderGrp(
        "scale_max_w2",
        label="Scale Max (Wave 2):",
        field=True,
        minValue=0.1,
        maxValue=20.0,
        value=load_pref("scale_max_w2", 1.2),
        precision=2,
        columnWidth3=(130, 50, 220),
        annotation="Maximum scale for grass near obstacles. Use smaller values for shorter grass around objects.",
    )

    cmds.floatSliderGrp(
        "proximity_field",
        label="Obstacle Density:",
        field=True,
        minValue=1.0,
        maxValue=5.0,
        value=load_pref("proximity", 1.0),
        precision=1,
        columnWidth3=(120, 50, 230),
        annotation="Density multiplier near obstacles. 1.0 = uniform density. Higher = more grass clustering around objects (foot traffic avoidance effect).",
    )

    cmds.intFieldGrp(
        "seed_field",
        label="Random Seed:",
        value1=load_pref("seed", 42),
        columnWidth2=(100, 100),
        annotation="Seed for random number generation. Same seed = same grass layout. Change to get a different distribution.",
    )

    cmds.floatSliderGrp(
        "gravity_weight_field",
        label="Gravity Weight:",
        field=True,
        minValue=0.0,
        maxValue=1.0,
        value=load_pref("gravity_weight", 0.75),
        precision=2,
        columnWidth3=(100, 50, 250),
        annotation="How much grass grows toward the sky vs perpendicular to the terrain. 0 = follows surface normal, 1 = always vertical, 0.75 = mostly vertical with slope awareness.",
    )

    cmds.setParent('..')  # exit columnLayout
    cmds.setParent('..')  # exit frameLayout

    # advanced section (expanded by default)
    cmds.frameLayout(label="Advanced", collapsable=True, collapse=False)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=3)

    cmds.floatFieldGrp(
        "noise_field",
        label="Noise Scale:",
        value1=load_pref("noise", 0.004),
        precision=4,
        columnWidth2=(100, 100),
        annotation="Spatial frequency of the wind noise pattern. Smaller = broader, smoother wind. Larger = more turbulent, localized gusts.",
    )

    cmds.floatFieldGrp(
        "time_field",
        label="Time Scale:",
        value1=load_pref("time", 0.008),
        precision=4,
        columnWidth2=(100, 100),
        annotation="Speed of wind animation over time. Smaller = slow, gentle swaying. Larger = fast, jittery movement.",
    )

    cmds.intSliderGrp(
        "octaves_field",
        label="Octaves:",
        field=True,
        minValue=1,
        maxValue=8,
        value=load_pref("octaves", 4),
        columnWidth3=(100, 50, 250),
        annotation="Layers of noise detail in the wind pattern. More octaves = more fine detail but slower computation. 3-5 is typical.",
    )

    cmds.floatSliderGrp(
        "persistence_field",
        label="Persistence:",
        field=True,
        minValue=0.1,
        maxValue=1.0,
        value=load_pref("persistence", 0.5),
        precision=2,
        columnWidth3=(100, 50, 250),
        annotation="How much each noise octave contributes. Lower = smoother wind. Higher = more chaotic, detailed turbulence.",
    )

    cmds.floatSliderGrp(
        "max_lean_field",
        label="Max Lean Angle:",
        field=True,
        minValue=0.0,
        maxValue=90.0,
        value=load_pref("max_lean", 30.0),
        precision=1,
        columnWidth3=(100, 50, 250),
        annotation="Maximum angle (degrees) that grass blades can lean in the wind. 30 = moderate lean, 60+ = very dramatic.",
    )

    cmds.floatSliderGrp(
        "min_distance_field",
        label="Min Distance:",
        field=True,
        minValue=1.0,
        maxValue=50.0,
        value=load_pref("min_distance", 5.0),
        precision=1,
        columnWidth3=(100, 50, 250),
        annotation="Minimum spacing between grass blades (scene units). Prevents overlapping blades. Increase for sparser look.",
    )

    cmds.floatSliderGrp(
        "cluster_falloff_field",
        label="Cluster Falloff:",
        field=True,
        minValue=0.1,
        maxValue=1.0,
        value=load_pref("cluster_falloff", 0.5),
        precision=2,
        columnWidth3=(100, 50, 250),
        annotation="How quickly grass density drops off away from obstacles. Lower = tighter clustering near edges. Higher = wider, softer gradient.",
    )

    cmds.floatSliderGrp(
        "edge_offset_field",
        label="Edge Offset:",
        field=True,
        minValue=1.0,
        maxValue=50.0,
        value=load_pref("edge_offset", 10.0),
        precision=1,
        columnWidth3=(100, 50, 250),
        annotation="Distance from obstacle edge where grass density peaks (scene units). Controls how far from objects the densest grass ring appears.",
    )

    cmds.setParent('..')  # exit columnLayout
    cmds.setParent('..')  # exit frameLayout

    # action buttons
    cmds.rowLayout(numberOfColumns=2, columnWidth2=(210, 210), height=40)

    cmds.button(
        label="Generate Grass",
        backgroundColor=(0.3, 0.6, 0.3),
        command=lambda *args: execute_grass_generation(),
        height=35,
        annotation="Generate grass on the terrain mesh with the current settings. Creates a MASH network with animated wind.",
    )

    cmds.button(
        label="Close",
        command=lambda *args: cmds.deleteUI(window),
        height=35,
        annotation="Close this window. Settings are saved automatically.",
    )

    cmds.setParent('..')  # exit rowLayout
    cmds.setParent('..')  # exit main columnLayout

    # show the window
    cmds.showWindow(window)
