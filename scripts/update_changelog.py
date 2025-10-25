"""Helper script for updating CHANGELOG.md.

This script assists with generating changelog entries from git commit history.
It analyzes commit messages and suggests categorized entries for the changelog.

Usage:
    python scripts/update_changelog.py --since-last-tag
    python scripts/update_changelog.py --since "2025-10-20"
    python scripts/update_changelog.py --link-issues

The script generates a draft that should be reviewed and edited for clarity
before adding to CHANGELOG.md.
"""
import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ChangelogGenerator:
    """Generates changelog entries from git commit history."""

    # Conventional Commits patterns
    COMMIT_PATTERN = re.compile(
        r'^(?P<type>\w+)(?:\((?P<scope>[^\)]+)\))?(?P<breaking>!)?: (?P<message>.+)$'
    )

    # Category mappings from Conventional Commits to Keep a Changelog
    CATEGORY_MAP = {
        'feat': 'Added',
        'add': 'Added',
        'fix': 'Fixed',
        'refactor': 'Changed',
        'perf': 'Changed',
        'docs': 'Changed',
        'style': 'Changed',
        'test': 'Changed',
        'build': 'Changed',
        'ci': 'Changed',
        'chore': 'Changed',
        'revert': 'Fixed',
    }

    # Issue/PR reference patterns
    ISSUE_PATTERN = re.compile(r'#(\d+)')

    def __init__(self, repo_root: str = "."):
        """Initialize changelog generator.

        Args:
            repo_root: Path to repository root directory
        """
        self.repo_root = Path(repo_root)
        self.entries: Dict[str, List[str]] = {
            'Added': [],
            'Changed': [],
            'Deprecated': [],
            'Removed': [],
            'Fixed': [],
            'Security': []
        }

    def get_commits_since(
        self,
        since: Optional[str] = None,
        since_tag: bool = False
    ) -> List[Tuple[str, str, str]]:
        """Get commits since a specific point.

        Args:
            since: Date string (e.g., "2025-10-20") or commit hash
            since_tag: If True, get commits since last git tag

        Returns:
            List of (hash, message, body) tuples
        """
        cmd = ['git', 'log', '--format=%H|||%s|||%b']

        if since_tag:
            # Get last tag
            try:
                result = subprocess.run(
                    ['git', 'describe', '--tags', '--abbrev=0'],
                    capture_output=True,
                    text=True,
                    cwd=self.repo_root,
                    check=True
                )
                last_tag = result.stdout.strip()
                cmd.append(f'{last_tag}..HEAD')
            except subprocess.CalledProcessError:
                print("No tags found, using all commits")
        elif since:
            cmd.append(f'--since={since}')

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.repo_root,
            check=True
        )

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('|||')
            if len(parts) >= 2:
                commit_hash = parts[0]
                message = parts[1]
                body = parts[2] if len(parts) > 2 else ""
                commits.append((commit_hash, message, body))

        return commits

    def parse_commit_message(self, message: str) -> Tuple[Optional[str], Optional[str], str, bool]:
        """Parse commit message using Conventional Commits format.

        Args:
            message: Commit message first line

        Returns:
            Tuple of (type, scope, description, is_breaking)
        """
        match = self.COMMIT_PATTERN.match(message)
        if match:
            return (
                match.group('type'),
                match.group('scope'),
                match.group('message'),
                bool(match.group('breaking'))
            )

        # Fallback: try to infer from message content
        message_lower = message.lower()
        if message_lower.startswith('add'):
            return ('feat', None, message, False)
        elif message_lower.startswith('fix'):
            return ('fix', None, message, False)
        elif message_lower.startswith('remove') or message_lower.startswith('delete'):
            return ('remove', None, message, False)
        elif message_lower.startswith('deprecate'):
            return ('deprecate', None, message, False)

        return (None, None, message, False)

    def extract_issue_refs(self, text: str, link_issues: bool = False, repo_url: Optional[str] = None) -> str:
        """Extract and optionally link issue references.

        Args:
            text: Text containing issue references (#123)
            link_issues: Whether to convert to markdown links
            repo_url: GitHub repository URL (e.g., https://github.com/user/repo)

        Returns:
            Text with linked or preserved issue references
        """
        if not link_issues or not repo_url:
            return text

        def replace_issue(match):
            issue_num = match.group(1)
            return f"[#{issue_num}]({repo_url}/issues/{issue_num})"

        return self.ISSUE_PATTERN.sub(replace_issue, text)

    def categorize_commits(
        self,
        commits: List[Tuple[str, str, str]],
        link_issues: bool = False,
        repo_url: Optional[str] = None
    ):
        """Categorize commits into changelog sections.

        Args:
            commits: List of (hash, message, body) tuples
            link_issues: Whether to link issue references
            repo_url: GitHub repository URL for issue linking
        """
        for commit_hash, message, body in commits:
            # Skip certain commit types
            if any(skip in message.lower() for skip in [
                'merge pull request',
                'merge branch',
                'bump version',
                'update changelog'
            ]):
                continue

            commit_type, scope, description, is_breaking = self.parse_commit_message(message)

            # Determine category
            if is_breaking:
                category = 'Changed'
                description = f"**BREAKING**: {description}"
            elif commit_type == 'remove' or 'remove' in message.lower():
                category = 'Removed'
            elif commit_type == 'deprecate' or 'deprecate' in message.lower():
                category = 'Deprecated'
            elif 'security' in message.lower() or commit_type == 'security':
                category = 'Security'
            else:
                category = self.CATEGORY_MAP.get(commit_type, 'Changed')

            # Extract issue references
            full_text = f"{description}\n{body}" if body else description
            entry = self.extract_issue_refs(full_text.split('\n')[0], link_issues, repo_url)

            # Add scope if present
            if scope:
                entry = f"**{scope}**: {entry}"

            # Add to appropriate category
            if entry not in self.entries[category]:  # Avoid duplicates
                self.entries[category].append(entry)

    def generate_markdown(self) -> str:
        """Generate markdown changelog section.

        Returns:
            Formatted markdown string
        """
        output = ["## [Unreleased]", ""]

        for category in ['Added', 'Changed', 'Deprecated', 'Removed', 'Fixed', 'Security']:
            if self.entries[category]:
                output.append(f"### {category}")
                for entry in self.entries[category]:
                    output.append(f"- {entry}")
                output.append("")

        return '\n'.join(output)

    def save_to_file(self, output_path: str = "CHANGELOG_DRAFT.md"):
        """Save generated changelog to file.

        Args:
            output_path: Path to save draft changelog
        """
        markdown = self.generate_markdown()

        output_file = self.repo_root / output_path
        output_file.write_text(markdown)

        print(f"Draft changelog saved to: {output_path}")
        print("\nReview and edit the draft, then manually add to CHANGELOG.md")


def get_repo_url() -> Optional[str]:
    """Get GitHub repository URL from git remote.

    Returns:
        Repository URL or None if not found
    """
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True
        )
        remote_url = result.stdout.strip()

        # Convert SSH URL to HTTPS
        if remote_url.startswith('git@github.com:'):
            remote_url = remote_url.replace('git@github.com:', 'https://github.com/')

        # Remove .git suffix
        if remote_url.endswith('.git'):
            remote_url = remote_url[:-4]

        return remote_url
    except subprocess.CalledProcessError:
        return None


def main():
    """Run changelog generator from command line."""
    parser = argparse.ArgumentParser(
        description='Generate changelog entries from git commits'
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--since-last-tag',
        action='store_true',
        help='Generate entries for commits since last git tag'
    )
    group.add_argument(
        '--since',
        type=str,
        help='Generate entries for commits since date (e.g., "2025-10-20")'
    )

    parser.add_argument(
        '--link-issues',
        action='store_true',
        help='Convert issue references (#123) to markdown links'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='CHANGELOG_DRAFT.md',
        help='Output file path (default: CHANGELOG_DRAFT.md)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = ChangelogGenerator()

    # Get commits
    if args.since_last_tag:
        print("Analyzing commits since last tag...")
        commits = generator.get_commits_since(since_tag=True)
    elif args.since:
        print(f"Analyzing commits since {args.since}...")
        commits = generator.get_commits_since(since=args.since)
    else:
        print("Analyzing commits from last 7 days...")
        commits = generator.get_commits_since(since='7 days ago')

    print(f"Found {len(commits)} commits")

    # Get repo URL for issue linking
    repo_url = get_repo_url() if args.link_issues else None
    if args.link_issues and repo_url:
        print(f"Linking issues to: {repo_url}")

    # Categorize and generate
    generator.categorize_commits(commits, link_issues=args.link_issues, repo_url=repo_url)
    generator.save_to_file(args.output)

    # Print preview
    print("\n" + "="*70)
    print("PREVIEW")
    print("="*70)
    print(generator.generate_markdown())
    print("="*70)
    print("\nIMPORTANT: This is a DRAFT. Please review and edit for:")
    print("  - User-focused language (what/why, not technical details)")
    print("  - Grouped related changes")
    print("  - Proper categorization")
    print("  - Clear descriptions")
    print("\nThen manually copy relevant sections to CHANGELOG.md")


if __name__ == "__main__":
    main()
