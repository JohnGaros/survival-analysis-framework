# GitHub Repository Setup Instructions

This guide will help you create a GitHub repository and push the Survival Analysis Framework.

## Quick Setup (3 Steps)

### Step 1: Create GitHub Repository

1. Go to **https://github.com/new**
2. Fill in the repository details:
   - **Repository name**: `survival-analysis-framework` (or your preferred name)
   - **Description**: "Customer churn prediction using survival analysis with Cox PH, Weibull AFT, GBSA, and RSF models"
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click **"Create repository"**

### Step 2: Add Remote and Push

After creating the repository, GitHub will show you commands. Use these commands from your terminal:

```bash
# Navigate to project directory (if not already there)
cd /Users/johngaros/Documents/Wemetrix/V326/survival/survival_framework

# Add the remote repository (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/survival-analysis-framework.git

# Or if you use SSH (recommended):
git remote add origin git@github.com:USERNAME/survival-analysis-framework.git

# Push the code
git push -u origin main
```

### Step 3: Verify

Visit your repository URL:
```
https://github.com/USERNAME/survival-analysis-framework
```

You should see all your files, including:
- METHODOLOGY.md and METHODOLOGY.docx
- Complete source code
- Tests and documentation
- Sample data and predictions

---

## Current Repository Status

✅ **Git initialized**: Repository ready to push
✅ **Initial commit created**: 231 files committed
✅ **Branch**: main
✅ **Commit hash**: 3c50c89

### What's Included in the Initial Commit

- **Core Framework**: Complete survival analysis implementation
- **Models**: Cox PH, Coxnet, Weibull AFT, GBSA, RSF
- **Preprocessing**: L2 regularization, missing value handling, variance filtering
- **Evaluation**: C-index, IBS, time-dependent AUC
- **Predictions**: Survival probabilities + expected survival time (RMST)
- **Testing**: Comprehensive test suite with >80% coverage
- **Documentation**:
  - METHODOLOGY.md/docx - Theoretical foundations
  - CLAUDE.md - Development guidelines
  - TESTING.md - Testing strategy
  - CODE_QUALITY.md - Quality standards
- **Sample Data**: 2000-record dataset for testing
- **Pre-commit Hooks**: Automated code quality checks
- **MLflow Tracking**: Experiment tracking with artifacts
- **Trained Models**: Pre-trained model files

---

## Alternative: Using GitHub CLI

If you prefer to use GitHub CLI (gh), first install it:

```bash
# macOS
brew install gh

# Authenticate
gh auth login

# Create and push repository
gh repo create survival-analysis-framework --public --source=. --remote=origin --push
```

---

## Recommended Repository Settings

After creating the repository, configure these settings on GitHub:

### Branch Protection (Settings → Branches)
- Protect the `main` branch
- Require pull request reviews
- Require status checks (if using CI/CD)

### Topics (Add repository topics)
Add these topics to make your repo discoverable:
- `survival-analysis`
- `machine-learning`
- `customer-churn`
- `cox-regression`
- `python`
- `scikit-survival`
- `mlflow`

### About Section
- Add description: "Customer churn prediction using survival analysis"
- Add website (if applicable)
- Add topics (as listed above)

---

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:

**For HTTPS:**
```bash
# Use personal access token
# Generate token at: https://github.com/settings/tokens
git remote set-url origin https://YOUR_TOKEN@github.com/USERNAME/survival-analysis-framework.git
```

**For SSH (recommended):**
```bash
# Set up SSH key first
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add this key to GitHub: Settings → SSH Keys

# Then use SSH remote
git remote set-url origin git@github.com:USERNAME/survival-analysis-framework.git
```

### Large File Warning

If you see warnings about large files:
```bash
# Check file sizes
find . -type f -size +50M

# Large model files are expected (RSF is ~78MB)
# These are gitignored in future commits via src/models/
```

### Push Rejected

If push is rejected:
```bash
# Make sure you're on the main branch
git branch

# Check remote URL
git remote -v

# Try force push (only for initial setup)
git push -u origin main --force
```

---

## Next Steps After Push

1. **Add README badges** (optional):
   - Build status
   - Coverage
   - Python version
   - License

2. **Set up CI/CD** (optional):
   - GitHub Actions for testing
   - Automated code quality checks

3. **Create releases**:
   ```bash
   git tag -a v1.0.0 -m "Initial release"
   git push origin v1.0.0
   ```

4. **Add collaborators** (if needed):
   - Settings → Collaborators → Add people

---

## Summary

Your survival analysis framework is ready to push to GitHub with:

- ✅ Complete implementation with 5 survival models
- ✅ Comprehensive documentation (markdown + Word format)
- ✅ Sample data and pre-trained models
- ✅ Testing infrastructure
- ✅ Pre-commit hooks for code quality
- ✅ MLflow experiment tracking
- ✅ Professional documentation and methodology

Simply create the repository on GitHub and run the `git remote add` and `git push` commands!
