"""maya grass plugin startup script.

this script runs when maya starts. it adds a shelf button for quick access
to the grass generator ui.

the button is added to a 'Custom' shelf. if the shelf doesn't exist, it is created.
"""

import maya.cmds as cmds
import maya.mel as mel


def add_grass_shelf_button():
    """add grass generator button to shelf."""
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


# defer execution until maya ui is ready
# (ui framework not initialized when userSetup.py runs)
cmds.evalDeferred(add_grass_shelf_button, lowestPriority=True)
