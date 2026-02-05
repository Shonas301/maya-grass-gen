#!/bin/zsh
# create a distributable zip of the maya grass plugin
# only includes the maya_grass_plugin/ directory (what users need to install)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$(dirname "$REPO_DIR")"
ZIP_NAME="maya-grass-gen.zip"

cd "$REPO_DIR"

# remove existing zip if present
rm -f "$OUTPUT_DIR/$ZIP_NAME"

# create zip with only the plugin directory
zip -r "$OUTPUT_DIR/$ZIP_NAME" maya_grass_plugin \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*.DS_Store"

echo ""
echo "Created: $OUTPUT_DIR/$ZIP_NAME"
ls -lh "$OUTPUT_DIR/$ZIP_NAME"
echo ""
echo "Contents:"
unzip -l "$OUTPUT_DIR/$ZIP_NAME" | head -30
