"""Analyze and optimize context file quality.

This script measures how well-optimized context files are for Claude Code
and provides recommendations for improvements.

Usage:
    python scripts/analyze_context.py

Output:
    - Console report with scores and recommendations
    - JSON file with detailed analysis (data/outputs/context_analysis.json)
"""
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import json


class ContextAnalyzer:
    """Analyzes context files for optimization opportunities."""

    def __init__(self, repo_root: str = "."):
        """Initialize context analyzer.

        Args:
            repo_root: Path to repository root directory
        """
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

        Checks for:
        - Required sections
        - Recent updates
        - File size (sweet spot: 500-3000 lines)
        - Code examples
        - Command documentation

        Returns:
            Score 0-100
        """
        claude_path = self.repo_root / "CLAUDE.md"
        score = 0

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
                    days_old = (datetime.now() - latest_date).days
                    self.recommendations.append({
                        "priority": "MEDIUM",
                        "file": "CLAUDE.md",
                        "issue": f"Last update was {days_old} days ago",
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
                "action": "Consider splitting or archiving old content"
            })

        # Check for code examples (10 points)
        code_blocks = content.count('```')
        if code_blocks >= 4:  # At least 2 code examples
            score += 10
        elif code_blocks > 0:
            score += 5
        else:
            self.recommendations.append({
                "priority": "LOW",
                "file": "CLAUDE.md",
                "issue": "No code examples found",
                "action": "Add code examples for common commands"
            })

        # Check for specific commands (10 points)
        if 'python' in content.lower() and ('pytest' in content.lower() or 'test' in content.lower()):
            score += 10

        self.scores["claude_md"] = round(score, 2)
        return score

    def analyze_readme(self) -> float:
        """Analyze README.md quality.

        Checks for:
        - Existence
        - Key sections (headers, installation, usage)
        - Code examples
        - Adequate length

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

        # Check for headers
        if "# " in content or "## " in content:
            score += 20

        # Check for installation instructions
        if "install" in content.lower():
            score += 20

        # Check for usage/examples
        if "usage" in content.lower() or "example" in content.lower():
            score += 20

        # Check for code blocks
        if "```" in content:
            score += 20
        else:
            self.recommendations.append({
                "priority": "MEDIUM",
                "file": "README.md",
                "issue": "No code examples",
                "action": "Add code examples for installation and usage"
            })

        # Check file is not too short
        if len(content) > 500:
            score += 20
        else:
            self.recommendations.append({
                "priority": "LOW",
                "file": "README.md",
                "issue": f"File is short ({len(content)} chars)",
                "action": "Expand with more details and examples"
            })

        self.scores["readme"] = round(score, 2)
        return score

    def analyze_module_readmes(self) -> float:
        """Check for module-level README files.

        Checks for:
        - src/README.md
        - tests/README.md
        - data/README.md

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

        self.scores["module_readmes"] = round(score, 2)
        return score

    def analyze_git_sync(self) -> float:
        """Check if context files are in sync with latest git changes.

        Compares CLAUDE.md modification time with last git commit.

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

            last_commit_str = result.stdout.strip()
            # Parse ISO date format
            last_commit = datetime.fromisoformat(
                last_commit_str.replace(' ', 'T', 1).rsplit(' ', 1)[0]
            )

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
                days_old = int(time_diff / 86400)
                score = 20
                self.recommendations.append({
                    "priority": "HIGH",
                    "file": "CLAUDE.md",
                    "issue": f"Not updated in {days_old} days",
                    "action": "Urgently update CLAUDE.md with recent changes"
                })

            self.scores["git_sync"] = round(score, 2)
            return score

        except Exception as e:
            print(f"Warning: Could not analyze git sync: {e}")
            self.scores["git_sync"] = 50
            return 50

    def analyze_skills(self) -> float:
        """Check for skills documentation.

        Counts skill files in .claude/skills/ directory.

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
            self.recommendations.append({
                "priority": "LOW",
                "file": ".claude/skills/",
                "issue": "No skill files",
                "action": "Document common workflows as skills"
            })
        elif skill_count < 3:
            score = 30
        elif skill_count < 5:
            score = 60
        else:
            score = 100

        self.scores["skills"] = round(score, 2)
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
        else:
            report.append("No recommendations - context files are well optimized!")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def save_report(self, output_path: str = "data/outputs/context_analysis.json"):
        """Save analysis results to JSON file.

        Args:
            output_path: Path to save JSON report
        """
        results = self.analyze_all()

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Run context analysis and print report."""
    analyzer = ContextAnalyzer()
    print(analyzer.generate_report())

    # Save to file
    output_path = "data/outputs/context_analysis.json"
    analyzer.save_report(output_path)
    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
