#!/bin/zsh
# cut a new release: validate, test, build zip, tag, push, create gh release
#
# usage:
#   ./scripts/release.sh 1.2.0
#   make release VERSION=1.2.0
#
# what it does:
#   1. validates semver format
#   2. checks for clean working tree
#   3. runs lint + typecheck + tests
#   4. builds the plugin zip
#   5. creates and pushes git tag vX.Y.Z
#   6. creates github release with the zip attached
#
# pass --dry-run to see what would happen without making changes

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DRY_RUN=false

usage() {
    echo "usage: $0 [--dry-run] <version>"
    echo ""
    echo "  version   semver string, e.g. 1.2.0 (the 'v' prefix is added automatically)"
    echo "  --dry-run show what would happen without making changes"
    echo ""
    echo "examples:"
    echo "  $0 1.1.0"
    echo "  $0 --dry-run 2.0.0"
    exit 1
}

info()  { echo "${GREEN}==>${NC} ${BOLD}$1${NC}"; }
warn()  { echo "${YELLOW}warning:${NC} $1"; }
fail()  { echo "${RED}error:${NC} $1" >&2; exit 1; }

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) usage ;;
        -*) fail "unknown flag: $1" ;;
        *) VERSION="$1"; shift ;;
    esac
done

[[ -z "${VERSION:-}" ]] && usage

# validate semver (loose: major.minor.patch with optional pre-release)
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
    fail "invalid version '$VERSION' — expected semver like 1.2.0 or 2.0.0-beta.1"
fi

TAG="v${VERSION}"

cd "$REPO_DIR"

# check gh cli is available
if ! command -v gh &>/dev/null; then
    fail "gh cli not found — install with: brew install gh"
fi

# check for clean working tree
if [[ -n "$(git status --porcelain)" ]]; then
    fail "working tree is dirty — commit or stash changes first"
fi

# check tag doesn't already exist
if git rev-parse "$TAG" &>/dev/null; then
    fail "tag $TAG already exists"
fi

info "releasing $TAG"

if $DRY_RUN; then
    echo ""
    warn "dry run — no changes will be made"
    echo ""
fi

# run checks
info "running checks (lint + typecheck + tests)..."
if $DRY_RUN; then
    echo "  would run: make all"
else
    make all
fi

# build zip
info "building plugin zip..."
ZIP_PATH="$REPO_DIR/maya-grass-gen.zip"
if $DRY_RUN; then
    echo "  would run: scripts/create_zip.sh (output to $ZIP_PATH)"
else
    # build zip into repo root instead of parent dir for cleaner release
    rm -f "$ZIP_PATH"
    cd "$REPO_DIR"
    zip -r "$ZIP_PATH" maya_grass_plugin \
        -x "*__pycache__*" \
        -x "*.pyc" \
        -x "*.DS_Store"
    echo "  built: $ZIP_PATH ($(du -h "$ZIP_PATH" | cut -f1 | xargs))"
fi

# tag
info "creating tag $TAG..."
if $DRY_RUN; then
    echo "  would run: git tag $TAG"
else
    git tag "$TAG"
fi

# push
info "pushing tag to origin..."
if $DRY_RUN; then
    echo "  would run: git push origin $TAG"
else
    git push origin "$TAG"
fi

# create github release
info "creating github release..."
if $DRY_RUN; then
    echo "  would run: gh release create $TAG $ZIP_PATH --title $TAG --generate-notes"
else
    gh release create "$TAG" "$ZIP_PATH" \
        --title "$TAG" \
        --generate-notes
fi

# cleanup zip from repo root
if ! $DRY_RUN; then
    rm -f "$ZIP_PATH"
fi

echo ""
info "done! release $TAG is live"
echo "  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$TAG"
