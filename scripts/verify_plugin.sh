#!/bin/zsh
# verify staged/packaged plugin artifact integrity

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_STAGE="$REPO_DIR/dist/stage/maya_grass_plugin"
VERSION_FILE="$REPO_DIR/src/maya_grass_gen/version.py"

usage() {
    echo "usage: $0 [--zip /path/to/maya-grass-gen.zip] [--version X.Y.Z]"
    exit 1
}

ZIP_PATH=""
PLUGIN_VERSION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip)
            shift
            [[ $# -eq 0 ]] && usage
            ZIP_PATH="$1"
            shift
            ;;
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

if [[ -z "$PLUGIN_VERSION" ]]; then
    PLUGIN_VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "$VERSION_FILE")"
fi

if [[ -z "$PLUGIN_VERSION" ]]; then
    echo "error: could not determine plugin version" >&2
    exit 1
fi

TARGET_DIR="$DEFAULT_STAGE"
TEMP_DIR=""

if [[ -n "$ZIP_PATH" ]]; then
    [[ ! -f "$ZIP_PATH" ]] && { echo "error: zip not found: $ZIP_PATH" >&2; exit 1; }
    TEMP_DIR="$(mktemp -d)"
    unzip -q "$ZIP_PATH" -d "$TEMP_DIR"
    TARGET_DIR="$TEMP_DIR/maya_grass_plugin"
fi

cleanup() {
    [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

[[ ! -d "$TARGET_DIR" ]] && { echo "error: plugin directory missing: $TARGET_DIR" >&2; exit 1; }
[[ ! -f "$TARGET_DIR/install.mel" ]] && { echo "error: install.mel missing" >&2; exit 1; }
[[ ! -f "$TARGET_DIR/scripts/userSetup.py" ]] && { echo "error: userSetup.py missing" >&2; exit 1; }
[[ ! -d "$TARGET_DIR/scripts/maya_grass_gen" ]] && { echo "error: runtime package missing" >&2; exit 1; }
[[ -L "$TARGET_DIR/scripts/maya_grass_gen" ]] && { echo "error: runtime package must not be a symlink" >&2; exit 1; }

if grep -q "__PLUGIN_VERSION__" "$TARGET_DIR/install.mel"; then
    echo "error: unresolved version placeholder in install.mel" >&2
    exit 1
fi
if grep -q "__PLUGIN_VERSION__" "$TARGET_DIR/scripts/userSetup.py"; then
    echo "error: unresolved version placeholder in userSetup.py" >&2
    exit 1
fi

if ! grep -q "$PLUGIN_VERSION" "$TARGET_DIR/install.mel"; then
    echo "error: install.mel is not stamped with version $PLUGIN_VERSION" >&2
    exit 1
fi
if ! grep -q "$PLUGIN_VERSION" "$TARGET_DIR/scripts/userSetup.py"; then
    echo "error: userSetup.py is not stamped with version $PLUGIN_VERSION" >&2
    exit 1
fi

STAGED_VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "$TARGET_DIR/scripts/maya_grass_gen/version.py")"
if [[ "$STAGED_VERSION" != "$PLUGIN_VERSION" ]]; then
    echo "error: staged python version ($STAGED_VERSION) does not match expected ($PLUGIN_VERSION)" >&2
    exit 1
fi

echo "[verify_plugin] OK: version=$PLUGIN_VERSION target=$TARGET_DIR"
