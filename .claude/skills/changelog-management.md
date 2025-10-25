# Changelog Management Skill

## Purpose

This skill guides the maintenance of CHANGELOG.md - a human-readable history of all notable changes to the project. The changelog helps developers, users, and stakeholders understand what has changed between versions, identify when bugs were introduced, and track the evolution of the project.

## Core Principle

**A changelog is for humans, not machines.** Every time you commit significant changes to the repository, you should update the CHANGELOG.md to document what changed and why it matters to users and developers.

## When to Use This Skill

**Trigger this skill after every git push that includes:**

1. ✅ **New features** - Added functionality
2. ✅ **Bug fixes** - Resolved issues
3. ✅ **Breaking changes** - API or behavior changes
4. ✅ **Performance improvements** - Speed or resource optimizations
5. ✅ **Deprecations** - Features being phased out
6. ✅ **Security fixes** - Vulnerability patches
7. ✅ **Refactoring** - Significant code reorganization
8. ✅ **Documentation updates** - Major doc improvements

**Skip for:**
- Typo fixes in comments
- Code formatting changes (black, isort)
- CI/CD configuration tweaks
- Minor comment updates

## Changelog Format

We follow the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format with these conventions:

### File Structure

```markdown
# Changelog

All notable changes to the Survival Analysis Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features that have been added

### Changed
- Changes to existing functionality

### Deprecated
- Features that will be removed in upcoming releases

### Removed
- Features that have been removed

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes

## [1.0.0] - 2025-10-25

### Added
- Initial release with Cox PH, Weibull AFT, RSF, and GBSA models
```

### Category Guidelines

**Added** - New features, capabilities, or files
```markdown
### Added
- Multi-level logging system with performance, warning, and debug logs
- Timer context manager for measuring execution time
- ProgressLogger class for tracking long-running operations
```

**Changed** - Changes to existing functionality
```markdown
### Changed
- Updated data preprocessing to include variance threshold filtering
- Improved LAPACK numerical stability with L2 regularization
- Refactored configuration system to centralize all hyperparameters
```

**Deprecated** - Features still available but will be removed
```markdown
### Deprecated
- `train_all_models(use_legacy_cv=True)` - Use new event-balanced CV instead
- Legacy artifact directory structure - Will be removed in v2.0.0
```

**Removed** - Features that have been deleted
```markdown
### Removed
- Legacy model storage in src/models/ (moved to run-type structure)
- 161 outdated markdown documentation files
- Deprecated `StratifiedCoxWrapper` with custom stratification
```

**Fixed** - Bug fixes
```markdown
### Fixed
- LAPACK error "leading minor of order X is not positive definite" (#42)
- Fold-aware time grid constraints causing IndexError in cross-validation
- Missing value imputation not handling edge cases with all-NaN columns
```

**Security** - Security patches
```markdown
### Security
- Updated scikit-survival to 0.22.2 to patch CVE-2024-XXXXX
- Added input validation to prevent SQL injection in data loading
```

## Writing Changelog Entries

### Best Practices

1. **User-focused language**: Explain *what* changed and *why* it matters
   - ❌ Bad: "Refactored train.py"
   - ✅ Good: "Improved training pipeline performance by 30% through parallel fold processing"

2. **Group related changes**: Combine related changes into single entries
   - ❌ Bad: Three separate entries for adding logging to main.py, train.py, validation.py
   - ✅ Good: "Added comprehensive logging system across all modules with performance tracking"

3. **Link to issues/PRs**: Include references when available
   ```markdown
   - Fixed LAPACK numerical stability errors in Cox models ([#42](https://github.com/user/repo/issues/42))
   ```

4. **Be specific**: Provide enough detail to understand the impact
   - ❌ Bad: "Updated configuration"
   - ✅ Good: "Centralized all hyperparameters in ModelConfig dataclass for easier tuning"

5. **Use present tense**: Describe what the change does, not what it did
   - ❌ Bad: "Added logging system"
   - ✅ Good: "Add logging system"

6. **Highlight breaking changes**: Make them obvious
   ```markdown
   ### Changed
   - **BREAKING**: `train_all_models()` now requires `run_type` parameter
   ```

### Entry Template

```markdown
## [Unreleased]

### Added
- [Brief description of new feature] ([#issue](link))
  - Additional context or sub-changes
  - Impact on users or workflow

### Changed
- [Description of change and why]
  - Migration guide if needed

### Fixed
- [Description of bug and fix] ([#issue](link))
```

## Workflow Integration

### Post-Commit Workflow

After every significant git push, follow this sequence:

1. **Update CHANGELOG.md**
   ```bash
   # Option 1: Manual update (recommended for quality)
   nano CHANGELOG.md  # Add entries under [Unreleased]

   # Option 2: Script-assisted (for first draft)
   python scripts/update_changelog.py --since-last-tag
   ```

2. **Review recent commits** to ensure nothing is missed
   ```bash
   git log --oneline --since="1 week ago"
   git log --oneline $(git describe --tags --abbrev=0)..HEAD
   ```

3. **Update CLAUDE.md** (via context-management skill)
   - Add to "Recent Updates" section
   - Update architecture if needed

4. **Commit changelog updates**
   ```bash
   git add CHANGELOG.md
   git commit -m "docs: Update CHANGELOG.md for recent changes"
   git push origin main
   ```

### Version Release Workflow

When cutting a new version:

1. **Move [Unreleased] to versioned release**
   ```markdown
   ## [Unreleased]

   ## [1.2.0] - 2025-10-25

   ### Added
   - Logging system with multi-level log files
   ...
   ```

2. **Create git tag**
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

3. **Add comparison links** at bottom of CHANGELOG.md
   ```markdown
   [Unreleased]: https://github.com/user/repo/compare/v1.2.0...HEAD
   [1.2.0]: https://github.com/user/repo/compare/v1.1.0...v1.2.0
   ```

## Automation Helper

The `scripts/update_changelog.py` script assists with changelog generation:

```bash
# Generate draft entries from recent commits
python scripts/update_changelog.py --since-last-tag

# Generate entries from specific date
python scripts/update_changelog.py --since "2025-10-20"

# Generate entries with issue linking
python scripts/update_changelog.py --link-issues
```

**Note**: Always review and edit script-generated entries for clarity and user focus.

## Examples from This Project

### Good Changelog Entries

```markdown
## [Unreleased]

### Added
- Comprehensive logging system with four log levels (main, performance, warnings, debug)
  - Real-time console output with color coding
  - Performance metrics logged separately for analysis
  - Automatic warning categorization (convergence, numerical, data, statistical)
  - Timer decorator and context manager for execution tracking
  - Created `src/survival_framework/logging_config.py` (348 lines)
  - Created `src/survival_framework/timing.py` (151 lines)

- Context management skill with automated quality analysis
  - Created `.claude/skills/context-management.md` with post-push workflow
  - Created `scripts/analyze_context.py` for context optimization scoring
  - Analyzes CLAUDE.md, README.md, module READMEs, git sync, and skills
  - Provides weighted score (0-100) with prioritized recommendations

### Changed
- Parameterized hardcoded filenames in context files for better maintainability
  - CLAUDE.md now uses `<path_to_csv_file>` placeholder instead of specific filename
  - Updated directory structure documentation to match actual layout
  - Integration tests show template pattern with concrete example

- Improved directory structure organization
  - Migrated from legacy `data/sample/` to run-type-based `data/inputs/{sample,production}/`
  - All outputs now in `data/outputs/{run_type}/` (models, artifacts, mlruns, logs)
  - Freed 382MB by removing duplicate legacy directories

### Removed
- 161 outdated markdown documentation files
  - Removed from git tracking to reduce repository size
  - Added `PHASE1_BENCHMARK.md` to .gitignore
  - Kept only actively maintained documentation

### Fixed
- LAPACK numerical stability errors in Cox proportional hazards models
  - Added L2 regularization (alpha=0.01) to prevent singular matrices
  - Implemented comprehensive missing value imputation with indicators
  - Added variance threshold filtering to remove constant features
  - Fold-aware time grid constraints for cross-validation
  - See `LAPACK_FIX_SUMMARY.md` for detailed technical changes
```

## Checklist for Changelog Updates

Before pushing changelog updates, verify:

- [ ] All significant changes from recent commits are documented
- [ ] Entries are in correct categories (Added, Changed, Fixed, etc.)
- [ ] Language is user-focused and clear
- [ ] Breaking changes are marked as **BREAKING**
- [ ] Issue/PR numbers are linked where applicable
- [ ] Date format is YYYY-MM-DD
- [ ] Entries are in reverse chronological order (newest first)
- [ ] Grammar and spelling are correct
- [ ] Related changes are grouped together
- [ ] Technical details are accurate but accessible

## Integration with Other Skills

### context-management.md
- **Trigger together**: After every git push
- **CHANGELOG.md**: Detailed history for humans (all changes)
- **CLAUDE.md Recent Updates**: Summary for Claude Code (last 5 major changes)
- **Difference**: CHANGELOG is exhaustive, CLAUDE.md is curated highlights

### logging-policy.md
- **Different purpose**: CHANGELOG is historical documentation, logging is runtime monitoring
- **No overlap**: CHANGELOG documents *what* changed, logs show *what* happens during execution

## Tips for Effective Changelogs

1. **Start on day one**: Don't wait to accumulate technical debt
2. **Write as you code**: Update CHANGELOG.md in the same commit when possible
3. **Think like a user**: What would someone upgrading want to know?
4. **Include migration guides**: For breaking changes, show before/after code
5. **Link to docs**: Reference updated documentation for complex changes
6. **Celebrate wins**: Use changelog to showcase improvements and milestones
7. **Be honest about bugs**: Transparently document what was broken and fixed

## References

- [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) - Standard format
- [Semantic Versioning](https://semver.org/spec/v2.0.0.html) - Version numbering
- [Conventional Commits](https://www.conventionalcommits.org/) - Commit message format
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github) - Release workflow

## Common Pitfalls to Avoid

1. ❌ **Git commit log dump**: Don't just copy commit messages
   - Commits are technical, changelog is user-focused

2. ❌ **Too granular**: Don't list every file change
   - Group related changes into coherent features

3. ❌ **Missing context**: Don't assume readers know the codebase
   - Explain why changes matter, not just what changed

4. ❌ **Irregular updates**: Don't let it fall behind
   - Update with every push, not quarterly

5. ❌ **Vague descriptions**: Don't use generic language
   - "Fixed bugs" → "Fixed race condition in parallel CV causing non-deterministic results"

## Maintenance Schedule

- **After every git push**: Update [Unreleased] section
- **Weekly review**: Ensure no changes were missed
- **Before releases**: Move [Unreleased] to versioned section
- **Monthly**: Review changelog for clarity and completeness
- **Quarterly**: Archive old versions if changelog becomes too large (move to CHANGELOG-archive.md)
