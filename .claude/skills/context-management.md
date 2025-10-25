# Context Management Skill

## Purpose

This skill guides the maintenance and optimization of context files (CLAUDE.md, README.md, etc.) to ensure Claude Code has the most relevant and up-to-date information for effective development work.

## Core Principle

**Context files should be living documents that evolve with the codebase.** After every significant change (especially git pushes), context files should be updated to reflect the current state, recent changes, and future directions.

## When to Use This Skill

**Trigger this skill proactively after:**
1. ✅ **Git push** - Always update context after pushing changes
2. ✅ **Major feature completion** - Document new capabilities
3. ✅ **Architecture changes** - Update structure documentation
4. ✅ **Breaking changes** - Warn about deprecated patterns
5. ✅ **Performance improvements** - Document optimizations
6. ✅ **Bug fixes** - Update known issues section

**Also trigger when:**
- User explicitly requests context update
- You notice context files are outdated (>1 week old modifications)
- New patterns or conventions are established
- Directory structure changes significantly

## Context Files to Maintain

### 1. CLAUDE.md (Primary Context File)

**Purpose**: Comprehensive guide for Claude Code to understand the project

**Required Sections:**

```markdown
# CLAUDE.md

## Project Overview
- Brief description (2-3 sentences)
- Main purpose and use case
- Key technologies and frameworks

## Recent Updates (Rolling Last 5)
**[Date]**: Brief summary of change
- Key additions or modifications
- Impact on workflow
- New patterns to follow

## Running Commands
### Setup and Installation
```bash
# Exact commands to set up environment
```

### Training Models
```bash
# All variations of main commands
```

### Testing
```bash
# Test commands and patterns
```

## Architecture
### Directory Structure
```
project/
├── key_directory/  # What it contains
└── another/        # Purpose
```

### Core Modules
- **module_name.py** - Purpose and key functions
- Include line numbers for critical functions

### Data Flow
1. Entry point → Processing → Output
2. Key transformations

## Development Standards
- Python version
- Code style (PEP 8, type hints, etc.)
- Testing requirements (>80% coverage)
- Docstring format (Google style)

## Important Implementation Notes
- Critical patterns to follow
- Common pitfalls to avoid
- Integration requirements

## Configuration System
- How to configure the application
- Available presets
- Customization patterns

## Known Issues and Workarounds
- Active bugs or limitations
- Temporary solutions
- Future improvements needed

## Git Workflow
- Branch strategy
- Commit message format
- PR requirements (if applicable)
```

### 2. README.md (User-Facing Documentation)

**Purpose**: User guide and project introduction

**Required Sections:**
- Project description and features
- Installation instructions
- Usage examples
- Configuration guide
- Contributing guidelines (if open source)
- License

**Keep synchronized with CLAUDE.md but:**
- More user-friendly tone
- Focus on "how to use" not "how it works"
- Include badges, screenshots if applicable

### 3. Module-Level README.md Files

**Locations:**
- `src/README.md` - Source code overview
- `tests/README.md` - Testing documentation
- `data/README.md` - Data organization
- `scripts/README.md` - Utility scripts

**Update when:**
- New modules added
- APIs change
- Directory structure evolves

### 4. Skills Documentation

**Location**: `.claude/skills/`

**Purpose**: Reusable workflows for specific tasks

**Update when:**
- New workflows established
- Best practices evolve
- Common tasks identified

## Context Optimization Function

### Implementation

Create `scripts/analyze_context.py`:

```python
"""Analyze and optimize context file quality.

This script measures how well-optimized context files are for Claude Code
and provides recommendations for improvements.
"""
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json


class ContextAnalyzer:
    """Analyzes context files for optimization opportunities."""

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.scores = {}
        self.recommendations = []

    def analyze_all(self) -> Dict:
        """Run all context analyses.

        Returns:
            Dictionary with scores, recommendations, and metrics
        """
        # Analyze each context file
        self.analyze_claude_md()
        self.analyze_readme()
        self.analyze_module_readmes()
        self.analyze_git_sync()
        self.analyze_skills()

        # Calculate overall score
        overall_score = self.calculate_overall_score()

        return {
            "overall_score": overall_score,
            "grade": self.get_grade(overall_score),
            "scores": self.scores,
            "recommendations": self.recommendations,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_claude_md(self) -> float:
        """Analyze CLAUDE.md quality.

        Returns:
            Score 0-100
        """
        claude_path = self.repo_root / "CLAUDE.md"
        score = 0
        max_score = 100

        if not claude_path.exists():
            self.recommendations.append({
                "priority": "HIGH",
                "file": "CLAUDE.md",
                "issue": "File does not exist",
                "action": "Create CLAUDE.md with project overview and recent updates"
            })
            self.scores["claude_md"] = 0
            return 0

        content = claude_path.read_text()

        # Check required sections (40 points)
        required_sections = [
            "Project Overview",
            "Running Commands",
            "Architecture",
            "Development Standards"
        ]
        sections_found = sum(1 for section in required_sections if section in content)
        score += (sections_found / len(required_sections)) * 40

        if sections_found < len(required_sections):
            missing = [s for s in required_sections if s not in content]
            self.recommendations.append({
                "priority": "HIGH",
                "file": "CLAUDE.md",
                "issue": f"Missing sections: {', '.join(missing)}",
                "action": f"Add sections: {', '.join(missing)}"
            })

        # Check for recent updates section (20 points)
        if "Recent Updates" in content or "RECENT UPDATES" in content.upper():
            score += 15
            # Check if updates are recent (within 30 days)
            date_pattern = r'\*\*(\d{4}-\d{2}-\d{2})\*\*'
            dates = re.findall(date_pattern, content)
            if dates:
                latest_date = max(datetime.strptime(d, "%Y-%m-%d") for d in dates)
                if datetime.now() - latest_date < timedelta(days=30):
                    score += 5
                else:
                    self.recommendations.append({
                        "priority": "MEDIUM",
                        "file": "CLAUDE.md",
                        "issue": f"Last update was {(datetime.now() - latest_date).days} days ago",
                        "action": "Add recent changes to Recent Updates section"
                    })
        else:
            self.recommendations.append({
                "priority": "MEDIUM",
                "file": "CLAUDE.md",
                "issue": "No Recent Updates section",
                "action": "Add Recent Updates section with last 5 changes"
            })

        # Check file size (20 points) - sweet spot is 500-3000 lines
        lines = content.split('\n')
        line_count = len(lines)
        if 500 <= line_count <= 3000:
            score += 20
        elif line_count < 500:
            score += (line_count / 500) * 20
            self.recommendations.append({
                "priority": "LOW",
                "file": "CLAUDE.md",
                "issue": f"File is short ({line_count} lines)",
                "action": "Add more detail to existing sections"
            })
        else:
            score += 10
            self.recommendations.append({
                "priority": "MEDIUM",
                "file": "CLAUDE.md",
                "issue": f"File is very long ({line_count} lines)",
                "action": "Consider splitting into multiple files or archiving old content"
            })

        # Check for code examples (10 points)
        code_blocks = content.count('```')
        if code_blocks >= 4:  # At least 2 code examples
            score += 10
        elif code_blocks > 0:
            score += 5

        # Check for specific commands (10 points)
        if 'python' in content.lower() and ('pytest' in content.lower() or 'test' in content.lower()):
            score += 10

        self.scores["claude_md"] = score
        return score

    def analyze_readme(self) -> float:
        """Analyze README.md quality.

        Returns:
            Score 0-100
        """
        readme_path = self.repo_root / "README.md"
        score = 0

        if not readme_path.exists():
            readme_path = self.repo_root / "readme.md"

        if not readme_path.exists():
            self.recommendations.append({
                "priority": "HIGH",
                "file": "README.md",
                "issue": "File does not exist",
                "action": "Create README.md with project description and usage"
            })
            self.scores["readme"] = 0
            return 0

        content = readme_path.read_text()

        # Check for key sections
        if "# " in content or "## " in content:
            score += 20

        if "install" in content.lower():
            score += 20

        if "usage" in content.lower() or "example" in content.lower():
            score += 20

        if "```" in content:
            score += 20

        # Check file is not too short
        if len(content) > 500:
            score += 20

        self.scores["readme"] = score
        return score

    def analyze_module_readmes(self) -> float:
        """Check for module-level README files.

        Returns:
            Score 0-100
        """
        expected_readmes = [
            "src/README.md",
            "tests/README.md",
            "data/README.md"
        ]

        found = sum(1 for path in expected_readmes
                   if (self.repo_root / path).exists())

        score = (found / len(expected_readmes)) * 100

        if found < len(expected_readmes):
            missing = [p for p in expected_readmes
                      if not (self.repo_root / p).exists()]
            self.recommendations.append({
                "priority": "LOW",
                "file": "Module READMEs",
                "issue": f"Missing: {', '.join(missing)}",
                "action": f"Create README files in: {', '.join(missing)}"
            })

        self.scores["module_readmes"] = score
        return score

    def analyze_git_sync(self) -> float:
        """Check if context files are in sync with latest git changes.

        Returns:
            Score 0-100
        """
        try:
            import subprocess

            # Get last commit date
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%cd', '--date=iso'],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            if result.returncode != 0:
                self.scores["git_sync"] = 50  # Can't determine
                return 50

            last_commit = datetime.fromisoformat(result.stdout.strip().replace(' ', 'T', 1).rsplit(' ', 1)[0])

            # Check CLAUDE.md modification time
            claude_path = self.repo_root / "CLAUDE.md"
            if not claude_path.exists():
                self.scores["git_sync"] = 0
                return 0

            claude_mtime = datetime.fromtimestamp(claude_path.stat().st_mtime)

            # Calculate time difference
            time_diff = (last_commit - claude_mtime).total_seconds()

            # Score based on how recently CLAUDE.md was updated relative to last commit
            if time_diff <= 0:  # Updated after last commit
                score = 100
            elif time_diff <= 3600:  # Within 1 hour
                score = 90
            elif time_diff <= 86400:  # Within 1 day
                score = 70
            elif time_diff <= 604800:  # Within 1 week
                score = 50
                self.recommendations.append({
                    "priority": "MEDIUM",
                    "file": "CLAUDE.md",
                    "issue": "Not updated in last week despite new commits",
                    "action": "Review recent commits and update CLAUDE.md"
                })
            else:  # More than 1 week
                score = 20
                self.recommendations.append({
                    "priority": "HIGH",
                    "file": "CLAUDE.md",
                    "issue": f"Not updated in {int(time_diff/86400)} days",
                    "action": "Urgently update CLAUDE.md with recent changes"
                })

            self.scores["git_sync"] = score
            return score

        except Exception as e:
            print(f"Warning: Could not analyze git sync: {e}")
            self.scores["git_sync"] = 50
            return 50

    def analyze_skills(self) -> float:
        """Check for skills documentation.

        Returns:
            Score 0-100
        """
        skills_dir = self.repo_root / ".claude" / "skills"

        if not skills_dir.exists():
            self.recommendations.append({
                "priority": "LOW",
                "file": ".claude/skills/",
                "issue": "No skills directory",
                "action": "Create .claude/skills/ for reusable workflows"
            })
            self.scores["skills"] = 0
            return 0

        # Count skill files
        skill_files = list(skills_dir.glob("*.md"))
        skill_count = len(skill_files)

        # Score based on number of skills
        if skill_count == 0:
            score = 0
        elif skill_count < 3:
            score = 30
        elif skill_count < 5:
            score = 60
        else:
            score = 100

        self.scores["skills"] = score
        return score

    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score.

        Weights:
        - CLAUDE.md: 40%
        - README.md: 25%
        - Module READMEs: 10%
        - Git sync: 20%
        - Skills: 5%

        Returns:
            Overall score 0-100
        """
        weights = {
            "claude_md": 0.40,
            "readme": 0.25,
            "module_readmes": 0.10,
            "git_sync": 0.20,
            "skills": 0.05
        }

        total = sum(self.scores.get(key, 0) * weight
                   for key, weight in weights.items())

        return round(total, 2)

    def get_grade(self, score: float) -> str:
        """Convert score to letter grade.

        Args:
            score: Score 0-100

        Returns:
            Letter grade A-F
        """
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def generate_report(self) -> str:
        """Generate human-readable report.

        Returns:
            Formatted report string
        """
        results = self.analyze_all()

        report = []
        report.append("=" * 70)
        report.append("CONTEXT OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append(f"Overall Score: {results['overall_score']}/100 (Grade: {results['grade']})")
        report.append(f"Generated: {results['timestamp']}")
        report.append("")

        report.append("Component Scores:")
        report.append("-" * 70)
        for component, score in results['scores'].items():
            bar_length = int(score / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            report.append(f"{component:20} {bar} {score:5.1f}/100")
        report.append("")

        if results['recommendations']:
            report.append("Recommendations:")
            report.append("-" * 70)
            # Group by priority
            by_priority = {"HIGH": [], "MEDIUM": [], "LOW": []}
            for rec in results['recommendations']:
                by_priority[rec['priority']].append(rec)

            for priority in ["HIGH", "MEDIUM", "LOW"]:
                if by_priority[priority]:
                    report.append(f"\n{priority} Priority:")
                    for rec in by_priority[priority]:
                        report.append(f"  [{rec['file']}]")
                        report.append(f"    Issue:  {rec['issue']}")
                        report.append(f"    Action: {rec['action']}")
                        report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def save_report(self, output_path: str = "context_analysis.json"):
        """Save analysis results to JSON file.

        Args:
            output_path: Path to save JSON report
        """
        results = self.analyze_all()
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Run context analysis and print report."""
    analyzer = ContextAnalyzer()
    print(analyzer.generate_report())

    # Save to file
    analyzer.save_report("data/outputs/context_analysis.json")
    print(f"\nDetailed report saved to: data/outputs/context_analysis.json")


if __name__ == "__main__":
    main()
```

## Context Update Workflow

### Automated Pre-commit Hook

**The project now includes an automated pre-commit hook that enforces context file updates!**

The hook (`scripts/pre_commit_context_check.py`) runs automatically before every commit and:
- ✅ Detects significant code changes (src/*.py, requirements.txt, etc.)
- ✅ Verifies CHANGELOG.md is updated for those changes
- ✅ Warns if CLAUDE.md hasn't been updated in >7 days
- ✅ Blocks commits if context files need updating
- ✅ Provides helpful guidance on what to update

**Setup** (one-time):
```bash
pip install pre-commit
pre-commit install
```

**What the hook checks:**

Significant changes requiring CHANGELOG.md update:
- Source code changes (`src/**/*.py`)
- Dependency changes (`requirements*.txt`)
- Package configuration (`setup.py`, `pyproject.toml`)

Documentation-only changes (no update required):
- Markdown files (`*.md`)
- Test files (`tests/`)
- Config files (`.pre-commit-config.yaml`, `.gitignore`)

**If commit is blocked:**
1. Update CHANGELOG.md under `[Unreleased]` section
2. Update CLAUDE.md if major change
3. Stage updated files: `git add CHANGELOG.md CLAUDE.md`
4. Retry commit

**Skip hook** (not recommended):
```bash
git commit --no-verify
```

### Manual Context Update Checklist

For manual updates or when making multiple related changes:

- [ ] **1. Update CHANGELOG.md** *(See changelog-management skill)*
  ```bash
  # Option 1: Manual update (recommended)
  nano CHANGELOG.md  # Add entries under [Unreleased]

  # Option 2: Script-assisted (for draft)
  python scripts/update_changelog.py --since-last-tag
  ```
  - Categorize changes: Added, Changed, Deprecated, Removed, Fixed, Security
  - Use user-focused language (what/why, not just technical details)
  - Link to issues/PRs if applicable
  - See `.claude/skills/changelog-management.md` for guidelines

- [ ] **2. Run Context Analysis** (optional)
  ```bash
  python scripts/analyze_context.py
  ```

- [ ] **3. Update CLAUDE.md**
  - Add entry to "Recent Updates" section (curated highlights, not all changes)
  - Update relevant sections if architecture changed
  - Update command examples if CLI changed
  - Add new patterns or conventions
  - **Note**: CLAUDE.md Recent Updates ≠ CHANGELOG.md
    - CLAUDE.md: Last 5 major changes for Claude Code context
    - CHANGELOG.md: Complete history for humans

- [ ] **4. Update README.md (if user-facing changes)**
  - Update usage examples
  - Add new features to feature list
  - Update installation if dependencies changed

- [ ] **5. Update Module READMEs (if structure changed)**
  - src/README.md for new modules
  - tests/README.md for new test patterns
  - data/README.md for data organization changes

- [ ] **6. Create/Update Skills (if new workflow)**
  - Document repeatable workflows
  - Capture new best practices

- [ ] **7. Commit Context Updates**
  ```bash
  git add CHANGELOG.md CLAUDE.md README.md src/README.md  # etc
  git commit -m "docs: update CHANGELOG and context files after [feature/change]"
  # Pre-commit hook will verify context is complete
  git push origin main
  ```

### CLAUDE.md Update Template

When updating CLAUDE.md after a push, use this template:

```markdown
## Recent Updates

**[YYYY-MM-DD]**: [Brief title of change]
- **Added**: [New features, modules, capabilities]
- **Changed**: [Modified behavior, refactored code]
- **Fixed**: [Bug fixes, corrections]
- **Impact**: [How this affects development workflow]
- **New patterns**: [Any new conventions to follow]

Example:
**[2025-10-25]**: Implemented comprehensive logging system
- **Added**: logging_config.py, timing.py with multi-level logging
- **Changed**: main.py and train.py now include extensive logging
- **Impact**: All runs now generate logs in data/outputs/{run_type}/logs/
- **New patterns**: Use @log_execution_time() decorator, Timer context manager
- **Files**: See .claude/skills/logging-policy.md for full guide
```

## Context Optimization Best Practices

### 1. Keep It Fresh
- Update within 24 hours of major changes
- Use "Recent Updates" as rolling changelog (keep last 5-10 entries)
- Archive old updates to separate file if CLAUDE.md gets too long

### 2. Be Specific
- Include actual commands, not placeholders
- Reference specific line numbers for critical functions
- Show concrete examples, not abstract descriptions

### 3. Highlight What's Important
- Put most recent and critical info at the top
- Use **bold** for key terms and changes
- Use code blocks for all commands and code snippets

### 4. Maintain Consistency
- Use consistent section headers across all context files
- Follow same markdown formatting style
- Keep similar level of detail across sections

### 5. Think Like a New Developer
- Would someone new understand this?
- Are commands copy-pasteable?
- Are dependencies clear?
- Are gotchas documented?

### 6. Optimize for Search
- Use clear section headers
- Include keywords that would be searched
- Cross-reference related sections

## Logging Context Optimization

Add to `src/survival_framework/logging_config.py`:

```python
def log_context_health(logger: logging.Logger):
    """Log context optimization metrics.

    Args:
        logger: Logger instance
    """
    from scripts.analyze_context import ContextAnalyzer

    analyzer = ContextAnalyzer()
    results = analyzer.analyze_all()

    logger.info("Context Health Check:")
    logger.info(f"  Overall Score: {results['overall_score']}/100 ({results['grade']})")

    for component, score in results['scores'].items():
        level = logging.INFO if score >= 70 else logging.WARNING
        logger.log(level, f"  {component}: {score}/100")

    if results['recommendations']:
        high_priority = [r for r in results['recommendations'] if r['priority'] == 'HIGH']
        if high_priority:
            logger.warning(f"  {len(high_priority)} HIGH priority recommendations")
```

## Integration with Git Hooks

Optional: Create `.git/hooks/post-commit` to remind about context updates:

```bash
#!/bin/bash
# Remind to update context files after commit

echo ""
echo "✓ Commit successful"
echo ""
echo "Reminder: Update context files if needed"
echo "  1. Run: python scripts/analyze_context.py"
echo "  2. Update CLAUDE.md Recent Updates section"
echo "  3. Review and apply recommendations"
echo ""
```

## Measuring Context Quality Metrics

Track these metrics over time:

1. **Overall Context Score** - Target: >80
2. **Git Sync Score** - How recent context is vs commits - Target: >90
3. **Recommendation Count** - Number of pending improvements - Target: <5 HIGH priority
4. **CLAUDE.md Size** - Keep between 500-3000 lines - Target: 1000-2000
5. **Skills Count** - Number of documented workflows - Target: >5

## Example Usage

### Basic Analysis
```bash
python scripts/analyze_context.py
```

### With Logging
```python
from survival_framework.logging_config import setup_logging
from scripts.analyze_context import ContextAnalyzer

logger = setup_logging(run_type="sample")
analyzer = ContextAnalyzer()
results = analyzer.analyze_all()

logger.info(f"Context Score: {results['overall_score']}/100")
for rec in results['recommendations']:
    if rec['priority'] == 'HIGH':
        logger.warning(f"Context issue: {rec['file']} - {rec['issue']}")
```

## Notes

- Context files are version controlled - track their evolution
- Regular updates prevent context drift
- Good context = more effective Claude Code assistance
- Treat context files as code documentation - keep them tested and current
- Context optimization is an investment in development velocity
