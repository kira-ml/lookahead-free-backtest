# Git Branching Strategy

This document outlines the Git branching strategy and workflow for the lookahead-free-backtest project.

## Branch Structure

### Permanent Branches

#### `main`
- **Purpose:** Production-ready code
- **Protection:** Protected branch, requires PR reviews
- **Merges from:** `staging` only
- **CI/CD:** Full test suite, deployment to production
- **Tagging:** All production releases are tagged (v1.0.0, v1.1.0, etc.)

#### `develop`
- **Purpose:** Integration branch for active development
- **Protection:** Protected branch, requires PR reviews
- **Merges from:** `feature/*`, `bugfix/*`, `experiment/*`
- **Merges to:** `staging`
- **CI/CD:** Full test suite, linting, causality audits
- **Usage:** Daily integration point for all development work

#### `staging`
- **Purpose:** Pre-production testing and validation
- **Protection:** Protected branch
- **Merges from:** `develop` only
- **Merges to:** `main`
- **CI/CD:** Full test suite, integration tests, performance benchmarks
- **Usage:** Final validation before production release

### Temporary Branches

#### `feature/*`
**Naming:** `feature/feature-name` (e.g., `feature/add-intraday-support`)

- **Purpose:** New features and enhancements
- **Branches from:** `develop`
- **Merges to:** `develop`
- **Lifetime:** Created for feature development, deleted after merge
- **Conventions:**
  - Use descriptive names: `feature/add-rolling-zscore`
  - Keep focused on a single feature
  - Update frequently from `develop` to avoid conflicts

**Example workflow:**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
# ... make changes ...
git add .
git commit -m "feat: Add my new feature"
git push -u origin feature/my-new-feature
# Create PR to develop
```

#### `bugfix/*`
**Naming:** `bugfix/bug-description` (e.g., `bugfix/fix-lookback-window`)

- **Purpose:** Non-critical bug fixes
- **Branches from:** `develop`
- **Merges to:** `develop`
- **Lifetime:** Short-lived, deleted after merge
- **Conventions:**
  - Reference issue numbers: `bugfix/issue-123-temporal-index`
  - Include tests that verify the fix

**Example workflow:**
```bash
git checkout develop
git pull origin develop
git checkout -b bugfix/fix-temporal-validation
# ... fix the bug ...
git add .
git commit -m "fix: Correct temporal validation logic"
git push -u origin bugfix/fix-temporal-validation
# Create PR to develop
```

#### `hotfix/*`
**Naming:** `hotfix/critical-issue` (e.g., `hotfix/lookahead-bias-v1.0.1`)

- **Purpose:** Critical production bug fixes
- **Branches from:** `main`
- **Merges to:** `main` AND `develop`
- **Lifetime:** Immediate, deleted after merge
- **Tagging:** Creates new patch version (v1.0.1, v1.0.2)
- **Priority:** Highest - bypasses normal flow for critical fixes

**Example workflow:**
```bash
git checkout main
git pull origin main
git checkout -b hotfix/critical-causality-bug
# ... fix critical bug ...
git add .
git commit -m "hotfix: Fix critical causality validator bug"
git push -u origin hotfix/critical-causality-bug
# Create PR to main (emergency review)
# After merge to main, also merge to develop
git checkout develop
git merge hotfix/critical-causality-bug
git push origin develop
```

#### `experiment/*`
**Naming:** `experiment/experiment-name` (e.g., `experiment/test-numba-acceleration`)

- **Purpose:** Experimental ML work, feature exploration, performance tests
- **Branches from:** `develop`
- **Merges to:** `develop` (if successful) or discarded
- **Lifetime:** Can be long-lived for research
- **Conventions:**
  - Document experiment goals in commit message
  - Track results in experiments/ directory
  - OK to have breaking changes

**Example workflow:**
```bash
git checkout develop
git pull origin develop
git checkout -b experiment/gpu-acceleration
# ... experiment with changes ...
git add .
git commit -m "experiment: Test GPU acceleration for feature compute"
git push -u origin experiment/gpu-acceleration
# If successful: Create PR to develop
# If failed: Document findings and delete branch
```

#### `release/*`
**Naming:** `release/vX.Y.0` (e.g., `release/v1.1.0`)

- **Purpose:** Release preparation and final testing
- **Branches from:** `develop`
- **Merges to:** `main` AND `develop`
- **Lifetime:** Created for release prep, deleted after release
- **Activities:**
  - Update version numbers
  - Update CHANGELOG.md
  - Final documentation updates
  - Bug fixes only (no new features)

**Example workflow:**
```bash
git checkout develop
git pull origin develop
git checkout -b release/v1.1.0
# Update version in setup.py
# Update CHANGELOG.md
git add .
git commit -m "chore: Prepare release v1.1.0"
git push -u origin release/v1.1.0
# Create PR to staging for final validation
# After staging tests pass, merge to main and tag
git checkout main
git merge release/v1.1.0
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin main --tags
# Also merge back to develop
git checkout develop
git merge release/v1.1.0
git push origin develop
```

## Workflow Diagrams

### Standard Feature Development Flow
```
develop → feature/my-feature → develop → staging → main
```

### Hotfix Flow
```
main → hotfix/critical-bug → main (tagged)
                           → develop (backport)
```

### Release Flow
```
develop → release/v1.1.0 → staging → main (tagged)
                                   → develop (backport)
```

## Commit Message Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat:** New feature
- **fix:** Bug fix
- **docs:** Documentation changes
- **style:** Code style changes (formatting, no logic change)
- **refactor:** Code refactoring (no feature change)
- **perf:** Performance improvements
- **test:** Adding or updating tests
- **chore:** Maintenance tasks (dependencies, build config)
- **experiment:** Experimental work (for experiment branches)

### Examples
```bash
feat(features): Add exponential weighted momentum feature

Implements EWM-based momentum calculation with configurable halflife.
Includes temporal validation and unit tests.

Closes #42

---

fix(core): Correct lookback window boundary in temporal index

The previous implementation included the as_of_date in the lookback
window, causing subtle lookahead bias. Now correctly excludes it.

---

experiment(performance): Test Numba JIT acceleration for rolling stats

Preliminary results show 3.5x speedup for rolling z-score computation.
Memory usage increased by 12%, needs optimization before production.
```

## Pull Request Guidelines

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Experiment (may not merge)

## Testing
- [ ] Unit tests added/updated
- [ ] Causality audit passes
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] CI/CD pipeline passes
```

### Review Requirements
- **develop:** 1 reviewer required
- **staging:** 1 reviewer + CI/CD pass
- **main:** 2 reviewers + full test suite + staging validation

## Branch Protection Rules

### GitHub Settings for Protected Branches

#### `main`
- ✅ Require pull request reviews (2 approvals)
- ✅ Require status checks to pass (CI/CD)
- ✅ Require branches to be up to date
- ✅ Require conversation resolution
- ✅ Require signed commits (recommended)
- ✅ Restrict who can push (maintainers only)

#### `develop`
- ✅ Require pull request reviews (1 approval)
- ✅ Require status checks to pass (CI/CD)
- ✅ Require branches to be up to date
- ✅ Require conversation resolution

#### `staging`
- ✅ Require pull request reviews (1 approval)
- ✅ Require status checks to pass (full test suite)
- ✅ Restrict who can merge (maintainers only)

## Quick Reference Commands

### Starting New Work
```bash
# Feature
git checkout develop && git pull origin develop
git checkout -b feature/my-feature

# Bugfix
git checkout develop && git pull origin develop
git checkout -b bugfix/fix-issue

# Hotfix
git checkout main && git pull origin main
git checkout -b hotfix/critical-fix

# Experiment
git checkout develop && git pull origin develop
git checkout -b experiment/test-idea
```

### Keeping Branch Updated
```bash
# Update from develop while on feature branch
git checkout feature/my-feature
git fetch origin
git merge origin/develop
# Or use rebase for cleaner history
git rebase origin/develop
```

### Cleaning Up After Merge
```bash
# Delete local branch
git branch -d feature/my-feature

# Delete remote branch (done automatically by GitHub after PR merge)
git push origin --delete feature/my-feature
```

### Listing Branches
```bash
# Local branches
git branch

# All branches (local + remote)
git branch -a

# Show last commit on each branch
git branch -v
```

## Best Practices

### Do's ✅
- **Commit frequently** with meaningful messages
- **Pull from develop/main regularly** to stay updated
- **Write tests** for new features and bug fixes
- **Keep PRs focused** - one feature/fix per PR
- **Update documentation** as you code
- **Run tests locally** before pushing
- **Use descriptive branch names** - future you will thank you
- **Delete branches after merge** to keep repository clean

### Don'ts ❌
- **Don't commit directly to main or develop**
- **Don't merge your own PRs** without review
- **Don't push broken code** - test first
- **Don't create mega-PRs** with 50+ files changed
- **Don't ignore CI/CD failures** - fix them
- **Don't force-push to shared branches**
- **Don't leave stale branches** hanging for months

## ML-Specific Considerations

### Data Versioning
- Use `data-vX` tags for major data updates (not branches)
- Document data lineage in experiments/ directory
- Keep processed feature versions in data/processed/

### Model Versioning
- Use Git tags for model versions: `model-v1.0.0`
- Store model artifacts externally (not in Git)
- Track model metadata in experiments/ directory

### Experiment Tracking
- Each experiment branch should update experiments/ with:
  - Experiment goals and hypothesis
  - Configuration used
  - Results and metrics
  - Decision: merge, iterate, or abandon

### Performance Benchmarks
- Include before/after benchmarks in PR description
- Run profiling before merging to main
- Document memory usage changes

## Emergency Procedures

### Reverting a Bad Merge
```bash
# Find the merge commit
git log --oneline

# Revert the merge
git revert -m 1 <merge-commit-hash>
git push origin main
```

### Recovering a Deleted Branch
```bash
# Find the commit where branch was deleted
git reflog

# Recreate branch from that commit
git checkout -b recovered-branch <commit-hash>
```

### Rolling Back Production
```bash
# Option 1: Revert specific commit
git revert <bad-commit-hash>
git push origin main

# Option 2: Hotfix with previous version
git checkout main
git checkout -b hotfix/rollback
git revert <bad-commit-hash>
git push -u origin hotfix/rollback
# Create PR and merge
```

## Automation Opportunities

### GitHub Actions Workflows
- Auto-delete merged feature branches
- Auto-label PRs based on file changes
- Auto-assign reviewers based on CODEOWNERS
- Run different test suites per branch type
- Auto-update CHANGELOG from commits

### Git Hooks
- Pre-commit: Run linters (black, flake8)
- Pre-push: Run unit tests
- Commit-msg: Validate commit message format

## Resources

- [Git Flow Cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
- [Semantic Versioning](https://semver.org/)

---

**Last Updated:** February 14, 2026  
**Maintained By:** ML Engineering Team
