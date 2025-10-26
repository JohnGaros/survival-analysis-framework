# Archive Management Skill

## Purpose

Automatically identify and archive completed task artifacts (planning documents, error analysis, one-off scripts) to keep the codebase clean and organized while preserving historical context.

## When to Use This Skill

### Proactive Triggers (Claude suggests)
- When noticing >5 old assessment files during file exploration
- After major milestone completion (e.g., post-recovery system implementation)
- When `.claude/assessments/` directory has >10 files

### User-Initiated Triggers
- User requests: "clean up old planning docs", "archive completed tasks", "organize old files"
- Periodic maintenance (monthly housekeeping)
- Before major releases or version bumps

## What Gets Archived

### Archival Candidates

1. **Assessment Files** (`.claude/assessments/*.md`)
   - Code reviews older than 30 days
   - Planning documents (*-plan.md, *-design.md)
   - Error analysis (*-fix-summary.md, *-error-analysis.md)
   - **Exception**: Keep most recent code review (last 7 days)

2. **One-off Scripts** (`scripts/`)
   - Recovery scripts for completed issues (e.g., `recover_pipeline.py`)
   - Benchmark scripts for completed experiments (e.g., `benchmark_production.py`)
   - Migration/fix scripts that are no longer needed
   - **Exception**: Keep actively maintained scripts (run-checks.sh, format-code.sh, etc.)

3. **Temporary Analysis Files** (root directory)
   - Ad-hoc analysis notebooks
   - Debugging scripts
   - Temporary data files

### What NEVER Gets Archived

- **Active documentation**: CLAUDE.md, README.md, CHANGELOG.md, METHODOLOGY.md
- **Active skills**: All files in `.claude/skills/`
- **Active agents**: All files in `.claude/agents/`
- **Source code**: Anything in `src/`
- **Tests**: Anything in `tests/`
- **Config files**: requirements.txt, pyproject.toml, .pre-commit-config.yaml
- **Recent files**: Modified within last 7 days (safety buffer)

## Archive Structure

```
.claude/archive/
├── assessments/
│   ├── 2025-10/
│   │   ├── code-review-2025-10-15.md
│   │   └── pipeline-resilience-plan.md
│   └── 2025-09/
│       └── auto-recovery-design.md
├── planning/
│   └── 2025-10/
│       └── feature-planning-docs.md
└── scripts/
    └── 2025-10/
        ├── recover_pipeline.py
        └── benchmark_production.py
```

Organization:
- Files grouped by type (assessments, planning, scripts)
- Subdirectories by year-month (YYYY-MM)
- Original filenames preserved
- README.md in each archive directory explaining contents

## Workflow

### Step 1: Scan for Candidates

```python
import os
from pathlib import Path
from datetime import datetime, timedelta

def find_archival_candidates():
    """Scan for files eligible for archival."""
    candidates = []
    now = datetime.now()
    cutoff_date = now - timedelta(days=30)
    recent_cutoff = now - timedelta(days=7)

    # Scan assessments
    assessments_dir = Path('.claude/assessments')
    if assessments_dir.exists():
        for file in assessments_dir.glob('*.md'):
            mtime = datetime.fromtimestamp(file.stat().st_mtime)

            # Skip recent files (safety buffer)
            if mtime > recent_cutoff:
                continue

            # Archive old planning/error docs
            if any(pattern in file.name for pattern in [
                '-plan', '-design', '-fix-summary', '-error-analysis'
            ]):
                candidates.append({
                    'path': file,
                    'type': 'planning',
                    'age_days': (now - mtime).days,
                    'reason': 'Planning/error document older than 7 days'
                })

            # Archive old code reviews (keep most recent)
            elif 'code-review-' in file.name and mtime < cutoff_date:
                candidates.append({
                    'path': file,
                    'type': 'assessment',
                    'age_days': (now - mtime).days,
                    'reason': 'Code review older than 30 days'
                })

    # Scan scripts
    scripts_dir = Path('scripts')
    if scripts_dir.exists():
        # One-off script patterns
        one_off_patterns = [
            'recover_', 'fix_', 'migrate_', 'benchmark_',
            'temp_', 'test_', 'debug_'
        ]

        for file in scripts_dir.glob('*.py'):
            if any(file.name.startswith(p) for p in one_off_patterns):
                mtime = datetime.fromtimestamp(file.stat().st_mtime)

                # Skip recent files
                if mtime > recent_cutoff:
                    continue

                candidates.append({
                    'path': file,
                    'type': 'script',
                    'age_days': (now - mtime).days,
                    'reason': 'One-off utility script'
                })

    return candidates
```

### Step 2: Present Candidates for Review

Display candidates to user with:
- File path
- File type (assessment/planning/script)
- Age in days
- Size
- Reason for archival

Example output:
```
Found 8 files eligible for archival:

ASSESSMENTS:
  .claude/assessments/code-review-2025-10-15.md
    Age: 11 days | Size: 12 KB | Reason: Code review older than 30 days

PLANNING:
  .claude/assessments/pipeline-resilience-plan.md
    Age: 8 days | Size: 15 KB | Reason: Planning document older than 7 days

  .claude/assessments/auto-recovery-design.md
    Age: 8 days | Size: 18 KB | Reason: Planning document older than 7 days

SCRIPTS:
  scripts/recover_pipeline.py
    Age: 8 days | Size: 4 KB | Reason: One-off utility script

Would you like to:
1. Archive all candidates
2. Review and select specific files
3. Cancel
```

### Step 3: Create Archive Structure

```bash
# Create archive directories
mkdir -p .claude/archive/assessments/$(date +%Y-%m)
mkdir -p .claude/archive/planning/$(date +%Y-%m)
mkdir -p .claude/archive/scripts/$(date +%Y-%m)

# Move files using git mv (preserves history)
git mv .claude/assessments/old-file.md .claude/archive/assessments/2025-10/
```

### Step 4: Create Archive README

For each archive directory, create/update README.md:

```markdown
# Archive: October 2025

## Contents

### Assessments
- `code-review-2025-10-15.md` - Code review from Oct 15
  - Archived: 2025-10-26
  - Reason: Older than 30 days

### Planning Documents
- `pipeline-resilience-plan.md` - Resilience improvements planning
  - Archived: 2025-10-26
  - Reason: Task completed, implementation done
  - Related commits: daa6298, a1b2c3d

## Retrieval

To restore a file:
```bash
git mv .claude/archive/assessments/2025-10/file.md .claude/assessments/
```

To view archive history:
```bash
git log --follow .claude/archive/assessments/2025-10/file.md
```
```

### Step 5: Commit Archive Changes

```bash
git add .claude/archive/
git commit -m "Archive completed task artifacts

Moved 8 files to .claude/archive/:
- 1 code review (>30 days old)
- 2 planning documents (tasks completed)
- 5 one-off scripts (no longer needed)

Files organized by type and month for easy retrieval.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## Safety Features

### Confirmation Required
- Always ask user before moving files
- Show full list of files to be archived
- Allow selective archival (not all-or-nothing)

### Dry Run Mode
- Option to preview what would be archived without moving files
- Show git mv commands that would be executed

### Rollback Support
```bash
# Undo last archive (if just committed)
git reset --soft HEAD~1
git restore --staged .

# Restore specific file
git mv .claude/archive/path/to/file.md original/location/
```

### Preserve Git History
- Always use `git mv` instead of `mv` to preserve file history
- Archive commits are atomic and reversible
- Each archive commit references related feature commits

## Integration with Other Skills

### Context Management
- Update CLAUDE.md if archived files were referenced
- Ensure no broken links in documentation
- Update `.claude/skills/README.md` if skill examples referenced archived files

### Changelog Management
- Don't archive CHANGELOG.md entries
- Add archive activity to CHANGELOG under maintenance section

## Examples

### Example 1: Archive Old Code Reviews

```
User: "Clean up old code reviews"

Claude: I found 3 code reviews older than 30 days:
- code-review-2025-09-15.md (41 days old, 12 KB)
- code-review-2025-09-22.md (34 days old, 15 KB)
- code-review-2025-10-01.md (25 days old, 10 KB)

I'll keep the most recent review (Oct 1) and archive the two older ones.

[Archives files, creates README, commits]

✓ Archived 2 code reviews to .claude/archive/assessments/2025-09/
```

### Example 2: Archive Planning Documents After Feature Complete

```
User: "Archive the recovery system planning docs"

Claude: Found these recovery-related planning documents:
- auto-recovery-design.md (8 days old, 18 KB) - Comprehensive design doc
- pipeline-resilience-plan.md (8 days old, 15 KB) - Implementation plan

Both are related to the completed automatic recovery feature (commit daa6298).

[Archives files with commit references, creates README]

✓ Archived 2 planning documents to .claude/archive/planning/2025-10/
✓ Preserved git history with related commit links
```

### Example 3: Proactive Suggestion

```
Claude: I notice there are 8 old assessment files and 5 one-off scripts in the
codebase. Would you like me to run the archive-management skill to clean them up?

User: "yes"

Claude: [Scans candidates, presents for review, archives with user confirmation]

✓ Archived 13 files to .claude/archive/
✓ Codebase cleaned, historical context preserved
```

## Maintenance

### Monthly Archival Checklist
1. Scan for candidates using criteria above
2. Review candidates (verify tasks are truly complete)
3. Archive files using git mv
4. Create/update archive README
5. Commit with descriptive message
6. Update CLAUDE.md if needed

### Annual Archive Review
- Every 6-12 months, review archive contents
- Consider compressing very old archives (>1 year)
- Verify no broken documentation links

## Best Practices

1. **Archive early, archive often** - Don't let completed task artifacts accumulate
2. **Preserve context** - Archive READMEs should explain why files were archived
3. **Use git mv** - Never lose history
4. **Be conservative** - When in doubt, don't archive (7-day buffer helps)
5. **Link to commits** - Archive READMEs should reference related feature commits
6. **Atomic commits** - One archive operation = one commit
7. **Reversible** - All archives can be easily restored if needed

## Quick Reference

### Invoke This Skill
```
User: "archive old planning docs"
User: "clean up completed tasks"
User: "organize old files"
```

### Manual Archive (for reference)
```bash
# Scan for candidates
find .claude/assessments -name "*.md" -mtime +30

# Create archive directory
mkdir -p .claude/archive/assessments/$(date +%Y-%m)

# Move file (preserving history)
git mv .claude/assessments/old-file.md .claude/archive/assessments/2025-10/

# Commit
git add .claude/archive/
git commit -m "Archive: old-file.md (task completed)"
```

### Restore Archived File
```bash
# Find file in archive
ls -R .claude/archive/

# Restore with git mv
git mv .claude/archive/path/to/file.md original/location/

# Commit restoration
git commit -m "Restore: file.md (needed for active work)"
```
