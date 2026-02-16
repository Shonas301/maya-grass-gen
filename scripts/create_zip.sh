#!/bin/zsh
# create a distributable zip of the maya grass plugin
# builds from a staged copy so release artifacts never depend on symlinks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$REPO_DIR/dist"
ZIP_NAME="maya-grass-gen.zip"
ZIP_PATH="$OUTPUT_DIR/$ZIP_NAME"
PLUGIN_VERSION=""

usage() {
    echo "usage: $0 [--version X.Y.Z]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            shift
            [[ $# -eq 0 ]] && usage
            PLUGIN_VERSION="$1"
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "error: unknown flag '$1'" >&2
            usage
            ;;
    esac
done

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

# remove existing zip if present
rm -f "$ZIP_PATH"

# build deterministic staged plugin and verify it
if [[ -n "$PLUGIN_VERSION" ]]; then
    "$SCRIPT_DIR/build_plugin.sh" --version "$PLUGIN_VERSION"
    "$SCRIPT_DIR/verify_plugin.sh" --version "$PLUGIN_VERSION"
else
    "$SCRIPT_DIR/build_plugin.sh"
    "$SCRIPT_DIR/verify_plugin.sh"
fi

# create zip from staged plugin directory
cd "$REPO_DIR/dist/stage"
zip -r "$ZIP_PATH" maya_grass_plugin \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*.DS_Store"

# verify zip artifact integrity
if [[ -n "$PLUGIN_VERSION" ]]; then
    "$SCRIPT_DIR/verify_plugin.sh" --zip "$ZIP_PATH" --version "$PLUGIN_VERSION"
else
    "$SCRIPT_DIR/verify_plugin.sh" --zip "$ZIP_PATH"
fi

echo ""
echo "Created: $ZIP_PATH"
ls -lh "$ZIP_PATH"
echo ""
echo "Contents:"
unzip -l "$ZIP_PATH" | sed -n "1,30p"
