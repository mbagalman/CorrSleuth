# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No        |

CorrSleuth is a data-analysis library: it does not open network connections,
execute user-supplied code, or read files beyond the pandas DataFrames you pass
it. The most likely security-relevant issues are in input handling (e.g.
pathological column names or data breaking report rendering) and in the supply
chain (dependencies, packaging, CI).

## Reporting a vulnerability

Please report suspected vulnerabilities privately — do not open a public issue.

- Preferred: [GitHub private vulnerability reporting](https://github.com/mbagalman/CorrSleuth/security/advisories/new)
- Or email: michael@paradoxresolution.com

Include a minimal reproduction if you can. You should receive an initial
response within a week. Fixes are released as patch versions and credited in
the changelog unless you prefer otherwise.
