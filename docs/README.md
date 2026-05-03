# CorrSleuth Documentation

User-facing documentation for the CorrSleuth package. Start with the
[main README](../README.md) for installation and quickstart.

## For analysts using CorrSleuth

- [Interpretation Guide](interpretation-guide.md) — what each diagnostic
  label means, when each metric can mislead, and what to do next. Covers
  every label with meaning, typical metric pattern, common examples,
  recommended next steps, and caveats.
- [Phase 4 Nonlinear Metrics Design Note](phase4-nonlinear-metrics-design-note.md)
  — why Chatterjee's ξ was chosen for `mode="deep"` over HSIC, MGC, MIC,
  and Hoeffding's D. Useful if you want to know *why* a particular
  dependence measure was added to the package.

## For contributors and maintainers

- [development/](development/) — internal planning, architecture, and the
  adoption ticket pack that drove the v0.x roadmap.
- [archive/](archive/) — historical artifacts (pre-release code reviews
  and their resolution logs).
