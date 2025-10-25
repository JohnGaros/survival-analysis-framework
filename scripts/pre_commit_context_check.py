#!/usr/bin/env python3
"""Pre-commit hook to verify context files are up to date.

This hook runs before each commit to ensure CLAUDE.md and CHANGELOG.md
are updated when significant code changes are made.

Exit codes:
    0: Context files are up to date or changes don't require update
    1: Context files need updating - commit blocked
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import re


class ContextChecker:
    """Checks if context files need updating before commit."""

    def __init__(self, repo_root: Optional[Path] = None):
        """Initialize context checker.

        Args:
            repo_root: Path to repository root (auto-detected if None)
        """
        if repo_root is None:
            repo_root = Path(__file__).parent.parent
        self.repo_root = repo_root
        self.claude_md = repo_root / "CLAUDE.md"
        self.changelog_md = repo_root / "CHANGELOG.md"
        self.warnings = []
        self.errors = []

    def get_staged_files(self) -> list[str]:
        """Get list of staged files for this commit.

        Returns:
            List of file paths staged for commit
        """
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            cwd=self.repo_root
        )
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().split('\n') if f]

    def is_significant_change(self, staged_files: list[str]) -> tuple[bool, str]:
        """Determine if staged changes are significant enough to require context update.

        Args:
            staged_files: List of staged file paths

        Returns:
            Tuple of (requires_update, reason)
        """
        # Ignore-only changes (docs, tests, configs that don't affect functionality)
        if not staged_files:
            return False, "No files staged"

        # Check for significant changes
        significant_patterns = [
            (r'src/.*\.py$', 'Source code changes'),
            (r'requirements.*\.txt$', 'Dependency changes'),
            (r'setup\.py$', 'Package configuration changes'),
            (r'pyproject\.toml$', 'Project configuration changes'),
        ]

        # Check for documentation-only changes
        doc_only = all(
            f.endswith('.md') or
            f.startswith('tests/') or
            f.startswith('.pre-commit') or
            f.startswith('.github/') or
            f.endswith('.gitignore')
            for f in staged_files
        )

        if doc_only:
            return False, "Documentation-only changes"

        # Check for significant changes
        for pattern, reason in significant_patterns:
            if any(re.match(pattern, f) for f in staged_files):
                return True, reason

        return False, "No significant changes detected"

    def check_claude_md_updated(self) -> bool:
        """Check if CLAUDE.md was recently modified.

        Returns:
            True if CLAUDE.md is up to date
        """
        if not self.claude_md.exists():
            self.errors.append("CLAUDE.md does not exist!")
            return False

        # Check if CLAUDE.md is in staged files
        staged = self.get_staged_files()
        if str(self.claude_md.relative_to(self.repo_root)) in staged:
            return True  # Being updated in this commit

        # Check last modification time
        mtime = datetime.fromtimestamp(self.claude_md.stat().st_mtime)
        age = datetime.now() - mtime

        if age > timedelta(days=7):
            self.warnings.append(
                f"CLAUDE.md hasn't been updated in {age.days} days. "
                "Consider adding recent changes to 'Recent Updates' section."
            )
            return True  # Warning only, don't block

        return True

    def check_changelog_updated(self, staged_files: list[str]) -> bool:
        """Check if CHANGELOG.md should be updated.

        Args:
            staged_files: List of staged file paths

        Returns:
            True if CHANGELOG.md is appropriately updated
        """
        if not self.changelog_md.exists():
            self.warnings.append(
                "CHANGELOG.md does not exist. Consider creating it."
            )
            return True  # Warning only

        # Check if CHANGELOG.md is in staged files
        if str(self.changelog_md.relative_to(self.repo_root)) in staged_files:
            return True  # Being updated in this commit

        # Get commit message to check if it's a docs commit
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            last_msg = result.stdout.strip().lower()

            # Skip changelog requirement for certain commit types
            skip_patterns = ['docs:', 'chore:', 'style:', 'test:']
            if any(last_msg.startswith(p) for p in skip_patterns):
                return True

        except Exception:
            pass

        # Check if significant code changes require changelog update
        significant, reason = self.is_significant_change(staged_files)
        if significant:
            self.errors.append(
                f"CHANGELOG.md not updated despite {reason.lower()}!\n"
                "  Add entry to [Unreleased] section following Keep a Changelog format.\n"
                "  Categories: Added, Changed, Deprecated, Removed, Fixed, Security"
            )
            return False

        return True

    def check_context_sync(self) -> bool:
        """Check if context files are in sync with code changes.

        Returns:
            True if context is in sync, False if updates needed
        """
        staged_files = self.get_staged_files()

        if not staged_files:
            return True  # No changes to check

        # Determine if this is a significant change
        significant, reason = self.is_significant_change(staged_files)

        if not significant:
            print(f"✓ Context check passed: {reason}")
            return True

        print(f"\n⚠️  Significant changes detected: {reason}")
        print(f"   Files: {', '.join(staged_files[:5])}")
        if len(staged_files) > 5:
            print(f"   ... and {len(staged_files) - 5} more")

        # Check both context files
        claude_ok = self.check_claude_md_updated()
        changelog_ok = self.check_changelog_updated(staged_files)

        # Print warnings
        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")

        # Print errors
        if self.errors:
            print("\n❌ Errors:")
            for error in self.errors:
                print(f"   - {error}")

        return claude_ok and changelog_ok and len(self.errors) == 0

    def print_help(self):
        """Print helpful message about updating context files."""
        print("\n" + "="*70)
        print("CONTEXT UPDATE REQUIRED")
        print("="*70)
        print("\nTo update context files before committing:\n")

        print("1. Update CHANGELOG.md:")
        print("   - Add entry under [Unreleased] section")
        print("   - Use appropriate category: Added, Changed, Fixed, etc.")
        print("   - Example:")
        print("     ### Added")
        print("     - New feature description\n")

        print("2. Update CLAUDE.md (if major change):")
        print("   - Add to 'Recent Updates' section")
        print("   - Update relevant sections (Architecture, Commands, etc.)\n")

        print("3. Stage updated files:")
        print("   git add CHANGELOG.md CLAUDE.md\n")

        print("4. Retry commit:")
        print("   git commit\n")

        print("Or skip this check (not recommended):")
        print("   git commit --no-verify\n")
        print("="*70)


def main():
    """Run pre-commit context check."""
    checker = ContextChecker()

    try:
        if checker.check_context_sync():
            sys.exit(0)  # Success
        else:
            checker.print_help()
            sys.exit(1)  # Block commit

    except Exception as e:
        print(f"\n⚠️  Context check failed with error: {e}")
        print("   Allowing commit to proceed...")
        sys.exit(0)  # Don't block on checker errors


if __name__ == "__main__":
    main()
