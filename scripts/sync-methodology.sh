#!/bin/bash
# Sync METHODOLOGY.docx from METHODOLOGY.md
#
# This script regenerates the Word document from the markdown source.
# It's called automatically by the pre-commit hook, but can also be
# run manually.

set -e

# Get script directory and repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

# Check if METHODOLOGY.md exists
if [ ! -f "METHODOLOGY.md" ]; then
    echo "Error: METHODOLOGY.md not found in repository root"
    exit 1
fi

# Check if python-docx is installed
if ! python -c "import docx" 2>/dev/null; then
    echo "Installing python-docx..."
    pip install python-docx
fi

# Run the conversion script
echo "Converting METHODOLOGY.md to METHODOLOGY.docx..."
python scripts/md_to_docx.py

# Check if the docx file was created/updated
if [ -f "METHODOLOGY.docx" ]; then
    echo "✓ METHODOLOGY.docx successfully synced"

    # Show file info
    ls -lh METHODOLOGY.docx

    # If in a git repository, stage the updated docx file
    if [ -d ".git" ] && git diff --quiet METHODOLOGY.md 2>/dev/null; then
        # METHODOLOGY.md has changes, stage the docx too
        git add METHODOLOGY.docx 2>/dev/null || true
        echo "✓ METHODOLOGY.docx staged for commit"
    fi
else
    echo "Error: Failed to create METHODOLOGY.docx"
    exit 1
fi
