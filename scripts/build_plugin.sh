#!/bin/zsh
# build a deterministic, self-contained plugin staging directory
# copies source package files instead of relying on symlinks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
STAGE_DIR="$REPO_DIR/dist/stage/maya_grass_plugin"
VERSION_FILE="$REPO_DIR/src/maya_grass_gen/version.py"

usage() {
    echo "usage: $0 [--version X.Y.Z]"
    exit 1
}

PLUGIN_VERSION=""
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

if [[ -z "$PLUGIN_VERSION" ]]; then
    PLUGIN_VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "$VERSION_FILE")"
fi

if [[ -z "$PLUGIN_VERSION" ]]; then
    echo "error: could not determine plugin version from $VERSION_FILE" >&2
    exit 1
fi

echo "[build_plugin] building maya_grass_plugin v$PLUGIN_VERSION"

rm -rf "$REPO_DIR/dist/stage"
mkdir -p "$STAGE_DIR"

# copy plugin scaffold (excluding symlinked runtime package)
rsync -a \
  --exclude 'scripts/maya_grass_gen' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$REPO_DIR/maya_grass_plugin/" "$STAGE_DIR/"

# copy runtime package from src so artifact never depends on symlinks
mkdir -p "$STAGE_DIR/scripts/maya_grass_gen"
rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$REPO_DIR/src/maya_grass_gen/" "$STAGE_DIR/scripts/maya_grass_gen/"

# stamp runtime python package version in staged artifact
cat > "$STAGE_DIR/scripts/maya_grass_gen/version.py" <<EOF
"""Project version metadata."""

__version__ = "$PLUGIN_VERSION"
EOF

# stamp version into runtime entrypoint files inside the staged artifact
sed -i.bak "s/__PLUGIN_VERSION__/$PLUGIN_VERSION/g" "$STAGE_DIR/install.mel"
sed -i.bak "s/__PLUGIN_VERSION__/$PLUGIN_VERSION/g" "$STAGE_DIR/scripts/userSetup.py"
rm -f "$STAGE_DIR/install.mel.bak" "$STAGE_DIR/scripts/userSetup.py.bak"

# sanity checks
if [[ -L "$STAGE_DIR/scripts/maya_grass_gen" ]]; then
    echo "error: staged runtime package is still a symlink" >&2
    exit 1
fi

if [[ ! -f "$STAGE_DIR/scripts/maya_grass_gen/__init__.py" ]]; then
    echo "error: missing staged runtime package" >&2
    exit 1
fi

echo "[build_plugin] staged at $STAGE_DIR"
