Maya Grass Plugin v1.0

INSTALLATION:
1. Drag 'install.mel' into Maya's viewport
2. Restart Maya
3. Find the Grass Generator button on the 'Custom' shelf

USAGE:
- Click the shelf button to open the Grass Generator UI
- Or in the Script Editor (Python):
    from maya_grass_gen import show_grass_ui
    show_grass_ui()

MANUAL INSTALLATION:
Copy 'maya_grass_plugin.mod' to:
  Windows: %USERPROFILE%/Documents/maya/modules/
  macOS: ~/maya/modules/
  Linux: ~/maya/modules/

Edit the path in the .mod file to point to this folder's location.

REQUIREMENTS:
- Maya 2022 or later (Python 3)
- opensimplex library (pip install opensimplex)
