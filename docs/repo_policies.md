<!-- MathJax configuration for MkDocs -->

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$','$$'], ['\\[','\\]']]
  }
};
</script>

<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<!-- Make all headers yellow -->

<style>
  h1, h2, h3, h4, h5, h6 {
    color: yellow;
  }
</style>

# Helios.Embed - Repository Policies

This document outlines the development, branching, and code review policies for the `Helios.Embed` repository.

---

## 1. Branching Strategy

*   **`main` Branch:** The `main` branch is the single source of truth. It must always be in a stable, releasable state. Direct pushes to `main` are disabled.
*   **Feature Branches:** All new work (features, bug fixes, documentation) must be done on a separate feature branch. Branch names should be descriptive, e.g., `feature/add-fused-kernel` or `fix/resolve-build-issue`.

## 2. Pull Request (PR) Policy

*   All code changes must be merged into the `main` branch via a Pull Request (PR).
*   Each PR must have a clear title and a description explaining the "what" and the "why" of the change.
*   **Review Required:** All PRs must be reviewed and approved by at least one official code owner as designated in the `.github/CODEOWNERS` file before merging.
*   **CI Checks Must Pass:** (Future-state) Once implemented, all Continuous Integration checks (build, tests, sanitizers) must pass before a PR can be merged.

## 3. Code Ownership

*   The official code owners for the entire repository are `@IRBSurfer` and `@Ashley-Kelly`.
*   This is formally defined in the `.github/CODEOWNERS` file.
