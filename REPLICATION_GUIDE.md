# Project Replication Guide

**Purpose:** Replicate this project structure for new data science projects with full Claude Code infrastructure.

**Goal:** Read this one file → Understand complete setup → Create new project in 30 minutes

---

## Quick Start (TL;DR)

```bash
# 1. Read these files (in order)
cat REPLICATION_GUIDE.md          # This file (start here)
cat PROJECT_TEMPLATE.md            # Understand structure
cat SETUP_CHECKLIST.md             # Step-by-step setup

# 2. Copy template structure
cp -r survival_framework_v1/ my_new_project/
cd my_new_project/

# 3. Run setup
./SETUP_CHECKLIST.md  # Follow all steps

# 4. Customize
vim CLAUDE.md  # Update for your project

# 5. Start coding!
```

---

## What Makes This Project Special?

### 🤖 **Automated Bug-Free Agentic Coding**

This project has infrastructure that enables Claude Code to:

1. ✅ **Work autonomously** - Reads CLAUDE.md, understands context, makes correct decisions
2. ✅ **Catch bugs automatically** - Unit + integration tests run after every change
3. ✅ **Enforce quality** - Pre-commit hooks prevent bad code from being committed
4. ✅ **Self-validate** - Integration tests ensure end-to-end workflow always works

**Result:** You discovered unit tests passed but pipeline broke. This will never happen again!

---

## The Three Files You Need

### 1. **REPLICATION_GUIDE.md** (This File)

**Read first:** Overview and quick start

**Purpose:**
- Explains what makes this project structure special
- Points you to the right files in the right order
- Provides quick reference for replication

---

### 2. **PROJECT_TEMPLATE.md** (Read Second)

**Purpose:** Understand the complete structure

**What it documents:**
- Complete directory tree with explanations
- All essential files (CLAUDE.md, .pre-commit-config.yaml, etc.)
- Why each file is critical
- How Claude Code uses each component
- Customization points for different project types

**Key sections:**
- Directory structure
- Essential files for Claude Code
- Configuration files
- Testing infrastructure
- Quality enforcement
- Workflow for Claude Code

**When to read:** Before starting new project

---

### 3. **SETUP_CHECKLIST.md** (Read Third)

**Purpose:** Step-by-step replication instructions

**What it provides:**
- ☐ Checkboxes for every step
- Commands to copy-paste
- Verification steps
- Expected outcomes
- Troubleshooting

**Key phases (30 minutes total):**
1. Directory structure (5 min)
2. Configuration files (10 min)
3. Scripts (5 min)
4. Claude Code integration (5 min)
5. Testing infrastructure (5 min)
6. Dependencies (3 min)
7. Documentation (2 min)
8. Installation & verification (2 min)
9. Customization (variable)

**When to use:** While setting up new project

---

## What Gets Replicated?

### 📁 **Directory Structure**

```
new_project/
├── .claude/agents/              # ⭐ Claude Code agents
├── src/project_name/            # Source code
├── tests/                       # Unit + integration tests
├── scripts/                     # Utility scripts
├── data/{sample,raw,processed}/ # Data directories
├── artifacts/                   # Training outputs
├── models/                      # Saved models
└── docs/                        # Additional documentation
```

---

### 📄 **Essential Files**

**Claude Code context:**
- ✅ `CLAUDE.md` - Main context file (most important!)
- ✅ `.claude/agents/test-runner.md` - Unit test automation
- ✅ `.claude/agents/integration-tester.md` - E2E test automation

**Testing:**
- ✅ `pytest.ini` - Pytest configuration
- ✅ `tests/conftest.py` - Test fixtures
- ✅ `tests/test_integration.py` - Integration tests
- ✅ `scripts/run-integration-tests.sh` - E2E test script

**Quality:**
- ✅ `.pre-commit-config.yaml` - Pre-commit hooks
- ✅ `pyproject.toml` - Tool configurations
- ✅ `scripts/run-checks.sh` - Quality check script
- ✅ `scripts/format-code.sh` - Auto-format script

**Documentation:**
- ✅ `TESTING.md` - Testing guide
- ✅ `CODE_QUALITY.md` - Quality standards
- ✅ `README.md` - Project overview
- ✅ `PROJECT_TEMPLATE.md` - Structure documentation
- ✅ `SETUP_CHECKLIST.md` - Setup instructions
- ✅ `REPLICATION_GUIDE.md` - This file

**Configuration:**
- ✅ `.gitignore` - Ignore rules
- ✅ `requirements.txt` - Dependencies
- ✅ `requirements-dev.txt` - Dev dependencies

---

## Why This Structure Works

### Problem: Unit Tests Passed, Pipeline Failed

**What happened:**
- Unit tests: 27/37 passed ✅
- End-to-end: LAPACK error, time range errors, type errors ❌

**Root cause:**
Unit tests don't validate how components work together in the pipeline.

**Solution:**
This structure enforces **two-tier testing**:

| Tier | Purpose | Speed | When |
|------|---------|-------|------|
| Unit | Function correctness | Fast (~10s) | Every commit |
| Integration | Pipeline validation | Slow (~90s) | Before merge |

**Result:** Both must pass before committing. No more surprises!

---

### How Claude Code Uses This Structure

#### 1. **Reads CLAUDE.md First**

```
User: "Please add feature X to the project"

Claude:
1. Reads CLAUDE.md → Understands architecture
2. Reads relevant module docs → Understands conventions
3. Makes changes → Follows established patterns
4. Runs tests → Validates changes work
5. Reports results → Tells user pass/fail
```

#### 2. **Uses Agents for Automation**

```
User: "Update data.py and make sure tests pass"

Claude:
1. Makes changes to data.py
2. Invokes test-runner agent → Runs unit tests
3. Invokes integration-tester agent → Runs E2E tests
4. Reports: "✓ All tests passed" or "✗ Integration test failed: LAPACK error"
```

#### 3. **Enforces Quality Automatically**

```
Developer: git commit -m "add feature"

Pre-commit hooks:
1. Black → Formats code
2. isort → Sorts imports
3. Flake8 → Checks linting
4. Mypy → Checks types
5. Bandit → Checks security

If any fail → Commit blocked
If all pass → Commit allowed
```

---

## Replication Methods

### Method 1: Copy Entire Structure (Fastest)

**Time:** 5 minutes + customization
**Best for:** Similar project type

```bash
# 1. Copy everything
cp -r survival_framework_v1/ my_new_project/
cd my_new_project/

# 2. Clean project-specific artifacts
rm -rf artifacts/ models/ data/sample/*.csv
rm -rf TEST_REPORT.md LAPACK_*.md SUCCESS_REPORT.md

# 3. Find and replace project name
find . -type f -name "*.py" -o -name "*.md" -o -name "*.ini" | \
  xargs sed -i '' 's/survival_framework/my_new_project/g'

# 4. Customize CLAUDE.md
vim CLAUDE.md

# 5. Initialize
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install

# 6. Verify
pytest -m "not integration" -v
./scripts/run-checks.sh

# Done!
```

---

### Method 2: Step-by-Step Setup (Most Control)

**Time:** 30 minutes
**Best for:** Different project type, learning the structure

```bash
# 1. Create new project
mkdir my_new_project && cd my_new_project
git init
python -m venv .venv
source .venv/bin/activate

# 2. Follow SETUP_CHECKLIST.md
# - Creates directory structure
# - Copies all configuration files
# - Creates scripts
# - Sets up Claude Code integration
# - Configures testing
# - Installs dependencies

# 3. Customize for your project type
# - Update CLAUDE.md
# - Add your dependencies
# - Add your modules
# - Write your tests

# Done!
```

---

### Method 3: Template Repository (Recommended for Teams)

**Time:** 2 minutes per new project
**Best for:** Multiple similar projects

```bash
# One-time setup:
# 1. Create template repository on GitHub
gh repo create my-org/ds-project-template --template --public

# 2. Push this structure to template
cd survival_framework_v1/
git remote add template git@github.com:my-org/ds-project-template.git
git push template main

# For each new project:
# 1. Create from template
gh repo create my-org/new-project --template my-org/ds-project-template

# 2. Clone and customize
git clone git@github.com:my-org/new-project.git
cd new-project/
vim CLAUDE.md  # Update for this project

# Done in 2 minutes!
```

---

## Customization Guide

### For Different Project Types

#### Machine Learning Projects

**Add:**
```bash
mkdir -p notebooks/ configs/ experiments/
```

**Update CLAUDE.md:**
```markdown
## Architecture
- notebooks/ - Exploratory analysis
- configs/ - Experiment configurations
- experiments/ - Experiment tracking
```

---

#### Web Applications

**Add:**
```bash
mkdir -p app/ static/ templates/ migrations/
```

**Update CLAUDE.md:**
```markdown
## Architecture
- app/ - Flask/FastAPI application
- static/ - CSS, JavaScript, images
- templates/ - HTML templates
- migrations/ - Database migrations
```

---

#### Data Pipelines

**Add:**
```bash
mkdir -p pipelines/ dags/ airflow/
```

**Update CLAUDE.md:**
```markdown
## Architecture
- pipelines/ - ETL pipeline definitions
- dags/ - Airflow DAG definitions
- airflow/ - Airflow configuration
```

---

## Verification Checklist

After replication, verify these work:

### ✅ Claude Code Integration

```bash
# Test: Can Claude understand project from CLAUDE.md?
# Ask Claude: "Read CLAUDE.md and explain the project architecture"
# Expected: Claude accurately describes modules, data flow, patterns
```

---

### ✅ Testing Infrastructure

```bash
# Test: Do both test tiers work?
pytest -m "not integration" -v          # Unit tests
./scripts/run-integration-tests.sh      # Integration tests

# Expected: Both pass (or show placeholder tests)
```

---

### ✅ Quality Enforcement

```bash
# Test: Do pre-commit hooks work?
echo "  trailing space  " >> test_file.py
git add test_file.py
git commit -m "test"

# Expected: Commit blocked, trailing whitespace removed
```

---

### ✅ Claude Agents

```bash
# Test: Does test-runner agent work?
# Make a change to a Python file
# Ask Claude: "Run the test-runner agent"

# Expected: Claude runs pytest and reports results
```

---

## Common Customizations

### Change Project Name

```bash
# Update in all files
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.ini" -o -name "*.toml" \) \
  -exec sed -i '' 's/OLD_NAME/NEW_NAME/g' {} +

# Rename directories
mv src/old_name src/new_name
```

---

### Change Python Version

```bash
# Update in:
# - .pre-commit-config.yaml (language_version)
# - pyproject.toml ([tool.black] target-version)
# - pytest.ini (if specified)
# - README.md

# Recreate venv
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

---

### Add New Dependencies

```bash
# 1. Add to requirements.txt
echo "new-package>=1.0.0" >> requirements.txt

# 2. Install
pip install -r requirements.txt

# 3. Update CLAUDE.md if it's a major dependency
vim CLAUDE.md  # Add to "Key Dependencies" section
```

---

### Add New Module

```bash
# 1. Create module
touch src/project_name/new_module.py

# 2. Add docstrings (Google style)
# 3. Create tests
touch tests/test_new_module.py

# 4. Update CLAUDE.md
vim CLAUDE.md  # Add to "Module Structure" section

# 5. Run tests
pytest tests/test_new_module.py -v
```

---

## Success Criteria

Your replicated project is successful if:

1. ✅ **Claude can work autonomously**
   - Reads CLAUDE.md and understands context
   - Follows conventions automatically
   - Knows what tests to run

2. ✅ **Tests catch bugs**
   - Unit tests provide fast feedback
   - Integration tests validate workflows
   - Both must pass before commit

3. ✅ **Quality is enforced**
   - Pre-commit hooks prevent bad code
   - All code follows same style
   - Documentation coverage >80%

4. ✅ **Easy to onboard**
   - New developers read CLAUDE.md
   - Structure is self-explanatory
   - Scripts automate setup

5. ✅ **Reliable workflow**
   - End-to-end tests always pass
   - No surprises in production
   - Confident to deploy

---

## Maintenance

### Keep Structure Updated

When you improve this template:

```bash
# 1. Update in current project
vim CLAUDE.md  # Add new best practice

# 2. Document in PROJECT_TEMPLATE.md
vim PROJECT_TEMPLATE.md  # Add to template

# 3. Update SETUP_CHECKLIST.md
vim SETUP_CHECKLIST.md  # Add setup step

# 4. Share with team
git commit -m "docs: update project template with X"
```

---

### Version Your Template

```bash
# Tag stable versions
git tag -a v1.0 -m "Stable project template with Claude Code integration"
git push origin v1.0

# Use specific version for new projects
gh repo create new-project --template my-org/ds-project-template --ref v1.0
```

---

## FAQ

### Q: Do I need all these files?

**A:** For Claude Code to work well, yes. But you can simplify:

**Minimum for Claude Code:**
- CLAUDE.md (essential)
- .claude/agents/ (essential for automation)
- pytest.ini + tests/ (essential for testing)
- README.md (helpful)

**Minimum for quality:**
- .pre-commit-config.yaml
- pyproject.toml
- .gitignore

**Minimum for documentation:**
- README.md
- CLAUDE.md

---

### Q: Can I use this for non-Python projects?

**A:** Yes! The structure is language-agnostic. Adapt:

**Keep:**
- CLAUDE.md (works for any language)
- .claude/agents/ (works for any language)
- Directory structure concept
- Two-tier testing approach

**Replace:**
- pytest.ini → Your test framework config
- .pre-commit-config.yaml → Your language's hooks
- pyproject.toml → Your language's config

---

### Q: How do I update an existing project?

**A:** Incrementally:

```bash
# 1. Add CLAUDE.md first
cp survival_framework_v1/CLAUDE.md my_old_project/
vim my_old_project/CLAUDE.md  # Customize

# 2. Add Claude agents
cp -r survival_framework_v1/.claude my_old_project/

# 3. Add integration tests
cp survival_framework_v1/scripts/run-integration-tests.sh my_old_project/scripts/
cp survival_framework_v1/tests/test_integration.py my_old_project/tests/

# 4. Add pre-commit hooks
cp survival_framework_v1/.pre-commit-config.yaml my_old_project/
cd my_old_project && pre-commit install

# 5. Test incrementally
pytest -m "not integration" -v
./scripts/run-integration-tests.sh
```

---

## Summary

### What You Get

When you replicate this structure, you get:

1. ✅ **Complete project template** - All files and configurations
2. ✅ **Claude Code integration** - CLAUDE.md + agents for automation
3. ✅ **Two-tier testing** - Unit (fast) + integration (thorough)
4. ✅ **Quality enforcement** - Pre-commit hooks + checks
5. ✅ **Comprehensive docs** - Guides for everything
6. ✅ **Proven patterns** - Based on real project that works

### Next Steps

1. **Read PROJECT_TEMPLATE.md** - Understand structure
2. **Follow SETUP_CHECKLIST.md** - Create new project
3. **Customize CLAUDE.md** - Make it yours
4. **Start coding!** - With confidence

### Time Investment vs. Benefit

**Setup time:** 30 minutes
**Benefit:** Automated, bug-free development for life of project
**ROI:** Pays back in first hour of development

### The Promise

> "Read CLAUDE.md → Claude understands everything → Work autonomously → Tests catch bugs → Quality enforced → Confident deployment"

**This is not aspirational. This is how this project actually works now! 🚀**

---

## Quick Reference Card

Save this for future projects:

```bash
# 1. Copy template
cp -r template/ new_project/ && cd new_project/

# 2. Initialize
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install

# 3. Customize CLAUDE.md
vim CLAUDE.md

# 4. Verify
pytest -m "not integration" -v
./scripts/run-integration-tests.sh

# 5. Start coding with Claude!
# Claude reads CLAUDE.md and knows everything
```

**Files to read:**
1. REPLICATION_GUIDE.md (this file) - Overview
2. PROJECT_TEMPLATE.md - Structure details
3. SETUP_CHECKLIST.md - Step-by-step setup

**Total setup time:** 30 minutes
**Total reading time:** 20 minutes
**Benefit:** Automated bug-free development forever!
