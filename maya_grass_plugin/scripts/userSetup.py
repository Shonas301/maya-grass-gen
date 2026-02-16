"""maya grass plugin startup script.

this script runs when maya starts. it adds a shelf button for quick access
to the grass generator ui and checks that required dependencies are available.

the button is added to a 'Custom' shelf. if the shelf doesn't exist, it is created.
"""

from maya import cmds, mel

MAYA_GRASS_PLUGIN_VERSION = "__PLUGIN_VERSION__"


def check_dependencies():
    """Check that required python dependencies are available."""
    missing = []

    try:
        import numpy  # noqa: ICN001
    except ImportError:
        missing.append("numpy")

    try:
        import opensimplex
    except ImportError:
        missing.append("opensimplex")

    if missing:
        deps = ", ".join(missing)
        cmds.confirmDialog(
            title="Grass Generator - Missing Dependencies",
            message=(
                f"The Grass Generator plugin is missing required dependencies:\n\n"
                f"  {deps}\n\n"
                f"To install, run this command in a terminal:\n\n"
                f"  mayapy -m pip install --user {' '.join(missing)}\n\n"
                f"On macOS, mayapy is typically at:\n"
                f"  /Applications/Autodesk/maya<version>/Maya.app/Contents/bin/mayapy\n\n"
                f"On Windows:\n"
                f"  C:\\Program Files\\Autodesk\\Maya<version>\\bin\\mayapy.exe\n\n"
                f"Restart Maya after installing."
            ),
            button=["OK"],
            defaultButton="OK",
            icon="warning",
        )
        cmds.warning(f"grass generator: missing dependencies: {deps}")


def add_grass_shelf_button():
    """Add grass generator button to shelf."""
    # get top shelf layout
    top_shelf = mel.eval("$tmpVar=$gShelfTopLevel")

    # get existing shelves
    shelves = cmds.tabLayout(top_shelf, query=True, childArray=True) or []
    shelf_name = "Custom"

    # create Custom shelf if it doesn't exist
    if shelf_name not in shelves:
        cmds.shelfLayout(shelf_name, parent=top_shelf)

    # check if button already exists (avoid duplicates on restart)
    shelf_buttons = cmds.shelfLayout(shelf_name, query=True, childArray=True) or []
    for btn in shelf_buttons:
        if cmds.shelfButton(btn, query=True, exists=True):
            ann = cmds.shelfButton(btn, query=True, annotation=True)
            if ann == "Open Grass Generator UI":
                # button already exists
                return

    # add button to shelf
    cmds.shelfButton(
        parent=shelf_name,
        annotation="Open Grass Generator UI",
        image1="commandButton.png",
        command="from maya_grass_gen import show_grass_ui; show_grass_ui()",
        sourceType="python",
    )

    print("grass generator shelf button added to 'Custom' shelf")


def startup():
    """Run all startup tasks."""
    print(
        "maya_grass_plugin "
        f"{MAYA_GRASS_PLUGIN_VERSION} startup entrypoint ({__file__})"
    )
    add_grass_shelf_button()
    check_dependencies()


# defer execution until maya ui is ready
# (ui framework not initialized when userSetup.py runs)
cmds.evalDeferred(startup, lowestPriority=True)
