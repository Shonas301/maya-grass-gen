Maya Grass Plugin v1.0

INSTALLATION:
1. Drag 'install.mel' into Maya's viewport
2. The installer will automatically:
   - Register the plugin module with Maya
   - Install numpy (requires internet connection)
   - Add a shelf button to the 'Custom' shelf
3. Restart Maya
4. Find the Grass Generator button on the 'Custom' shelf

If numpy fails to install automatically, run this in a terminal:
  macOS:   /Applications/Autodesk/maya<version>/Maya.app/Contents/bin/mayapy -m pip install --user numpy
  Windows: "C:\Program Files\Autodesk\Maya<version>\bin\mayapy.exe" -m pip install --user numpy
  Linux:   /usr/autodesk/maya<version>/bin/mayapy -m pip install --user numpy

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

BUNDLED DEPENDENCIES:
- opensimplex 0.4.5 (included in vendor/ - no install needed)

REQUIREMENTS:
- Maya 2022 or later (Python 3)
- numpy (installed automatically by the installer)
