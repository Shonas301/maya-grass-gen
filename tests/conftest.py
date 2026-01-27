"""shared test configuration - mock maya modules before any imports."""

import sys
from unittest.mock import MagicMock

# mock maya modules before any test imports maya_grass_gen
# (maya_grass_gen.__init__ imports ui.py which imports maya.cmds)
_mock_maya = MagicMock()
_mock_cmds = MagicMock()
_mock_maya.cmds = _mock_cmds

for mod_name in ["maya", "maya.cmds", "maya.api", "maya.api.OpenMaya",
                 "maya.OpenMaya", "maya.standalone", "MASH", "MASH.api"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
