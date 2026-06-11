#!/usr/bin/env bash
set -euo pipefail

PART="${1:-patch}"

if ! git diff --stat --exit-code; then
    echo "error: working tree has uncommitted changes; aborting." >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "master" ] && [ "$BRANCH" != "main" ]; then
    echo "warning: releasing from branch '$BRANCH' (not master/main)"
fi

echo "==> bumpversion $PART"
rtk bumpversion "$PART" --tag --verbose --commit 2>/dev/null || {
    echo "NOTE: bumpversion not installed. Install with: pip install bumpversion"
    echo "Manual steps: update version in pyproject.toml, commit, tag, push."
    exit 0
}

echo "==> push commit + tags"
rtk git push
rtk git push --tags

echo "==> build"
rtk python -m build

echo "==> gh release"
TAG=$(git describe --tags --abbrev=0)
rtk gh release create "$TAG" --generate-notes

echo "==> done: $TAG released"
