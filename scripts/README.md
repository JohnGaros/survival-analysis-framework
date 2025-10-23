# Utility Scripts

This directory contains helper scripts for development, code quality management, and documentation.

## Documentation Scripts

### md_to_docx.py

Converts `METHODOLOGY.md` to `METHODOLOGY.docx` with proper formatting.

**Features:**
- Preserves headers, bullets, numbered lists
- Formats code blocks with monospace font and gray background
- Converts tables to Word tables
- Applies consistent styling

**Usage:**
```bash
python scripts/md_to_docx.py
```

**Dependencies:**
- `python-docx>=1.1.0` (installed via `requirements-dev.txt`)

### sync-methodology.sh

Convenience script to regenerate METHODOLOGY.docx from METHODOLOGY.md.

**Usage:**
```bash
./scripts/sync-methodology.sh
```

**When to use:**
- After manually editing METHODOLOGY.md
- To verify the pre-commit hook is working
- When distributing documentation updates

**Automatic Sync:**

A pre-commit hook automatically runs this conversion when `METHODOLOGY.md` is modified. Configure in `.pre-commit-config.yaml`:

```yaml
- id: sync-methodology-docx
  name: Sync METHODOLOGY.docx from METHODOLOGY.md
  entry: python scripts/md_to_docx.py
  files: ^METHODOLOGY\.md$
```

## Code Quality Scripts

### setup-pre-commit.sh

Sets up pre-commit hooks for automated code quality checks.

**Usage:**
```bash
./scripts/setup-pre-commit.sh
```

**What it does:**
1. Installs pre-commit if not present
2. Installs git hooks to `.git/hooks/`
3. Updates hooks to latest versions
4. Runs initial check on all files

**Requirements:**
- Git repository initialized
- Python and pip installed
- requirements-dev.txt dependencies installed

### run-checks.sh

Runs all code quality checks manually.

**Usage:**
```bash
./scripts/run-checks.sh
```

**Checks performed:**
1. Black (formatting)
2. isort (import sorting)
3. Flake8 (linting)
4. Mypy (type checking)
5. Pylint (code analysis)
6. Bandit (security)
7. Interrogate (docstring coverage)
8. Pytest (tests + coverage)

**Exit codes:**
- 0: All checks passed
- 1: One or more checks failed

### format-code.sh

Auto-formats code using black and isort.

**Usage:**
```bash
./scripts/format-code.sh
```

**What it does:**
1. Formats all Python files with black
2. Sorts imports with isort
3. Displays summary of changes

**Safe to use:** Both tools only modify formatting, not logic

## Examples

### Before Committing

```bash
# Format code
./scripts/format-code.sh

# Run all checks
./scripts/run-checks.sh

# If all pass, commit
git add .
git commit -m "feat: add new feature"
```

### CI/CD Integration

```bash
# In CI pipeline
./scripts/run-checks.sh || exit 1
```

### Quick Fix

```bash
# Auto-fix formatting issues
./scripts/format-code.sh

# Check if it fixed everything
./scripts/run-checks.sh
```

## Troubleshooting

### Permission Denied

```bash
chmod +x scripts/*.sh
```

### Script Not Found

```bash
# Run from project root
cd /path/to/survival_framework_v1
./scripts/setup-pre-commit.sh
```

### Pre-commit Installation Fails

```bash
pip install --upgrade pre-commit
pre-commit clean
./scripts/setup-pre-commit.sh
```

## Notes

- All scripts should be run from the project root directory
- Scripts are bash scripts and require bash shell
- For Windows, use Git Bash or WSL
- Scripts are safe to run multiple times
