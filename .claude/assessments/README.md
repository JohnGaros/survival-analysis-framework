# Code Review Assessments

This directory contains code review assessments performed on the survival analysis framework.

## Purpose

Code review assessments provide:
- Comprehensive analysis of code quality
- Identification of bugs, security issues, and technical debt
- Prioritized feedback (Critical/Warnings/Suggestions)
- Tracking of improvements over time
- Documentation of decisions and trade-offs

## Assessment Format

Each assessment is stored as a dated Markdown file:
- `code-review-YYYY-MM-DD.md` - Full codebase reviews
- `feature-review-YYYY-MM-DD-feature-name.md` - Feature-specific reviews
- `security-review-YYYY-MM-DD.md` - Security-focused audits

## Assessment Structure

Each review includes:

### Executive Summary
- Overall quality score
- Key strengths and weaknesses
- Critical issues count

### Issue Categories
- 🔴 **Critical Issues** - Must fix before production
- 🟡 **Warnings** - Should fix soon
- 🟢 **Suggestions** - Nice to have improvements

### Detailed Analysis
- File-by-file breakdown
- Specific line numbers and code examples
- Impact assessment
- Recommended fixes

### Positive Findings
- Well-implemented features
- Good practices to maintain
- Exemplary code sections

### Recommendations
- Immediate action items
- High priority improvements
- Nice-to-have enhancements

## Using Assessments

### For Developers
- Review assessments before starting work
- Reference when fixing issues
- Use as checklist for new features
- Track progress over time

### For Code Reviews
- Compare against latest assessment
- Ensure new code doesn't reintroduce issues
- Reference standards and best practices
- Validate fixes address root causes

### For Project Planning
- Prioritize technical debt
- Estimate fix time
- Track quality metrics
- Plan refactoring sprints

## Running Code Reviews

### Manual Review
Request a code review from Claude Code:

```
> Review the codebase for quality, correctness, and best practices
> Focus on [specific area] and check for [specific issues]
```

### Automated Review
The code-reviewer agent can be invoked explicitly:

```
> Use the code-reviewer agent to review recent changes
```

Or it will run automatically after code modifications (if configured as proactive).

### Pre-commit Checks
The pre-commit hook (`scripts/pre_commit_context_check.py`) runs automatically to ensure:
- Context files are updated
- Significant changes are documented
- Standards are maintained

## Assessment History

| Date | Type | Scope | Score | Critical Issues | Status |
|------|------|-------|-------|----------------|--------|
| 2025-10-26 | Full codebase | Recent automation improvements | B+ (85/100) | 3 files | Open |

## Best Practices

1. **Review after significant changes** - Don't wait for problems to accumulate
2. **Document decisions** - Explain why certain issues aren't fixed
3. **Track over time** - Compare assessments to measure improvement
4. **Act on criticals** - Always fix critical issues before merging
5. **Share with team** - Use assessments for knowledge sharing

## Related Documentation

- [Code-reviewer Agent](../ agents/code-reviewer.md) - Automated code review agent
- [Code Quality Guide](../../CODE_QUALITY.md) - Project code quality standards
- [Testing Guide](../../TESTING.md) - Testing requirements and practices
- [CHANGELOG.md](../../CHANGELOG.md) - Project change history

---

*Assessments are living documents. Update them as issues are resolved and new patterns emerge.*
