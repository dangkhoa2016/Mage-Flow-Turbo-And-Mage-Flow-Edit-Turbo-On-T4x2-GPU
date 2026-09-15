# Security Policy

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

## Supported versions

Security fixes are applied to the current `main` branch and, after the first stable release, to the latest supported stable release when practical. Pre-release snapshots and historical commits are not guaranteed to receive backports.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability, exposed credential, authentication bypass, unsafe file handling, remote-code-execution path, dependency compromise, or other security-sensitive finding.

Use GitHub private vulnerability reporting when it is enabled for the repository. If that channel is unavailable, email `i.am@dangkhoa.dev` with:

- a concise description of the issue;
- affected component and revision;
- reproduction steps or proof of concept;
- expected security impact;
- any suggested mitigation, if known.

Do not include real production credentials or third-party private data in a report.

## Response expectations

A report will be acknowledged and triaged as availability permits. Confirmed issues will be handled through a coordinated fix and disclosure process. Public disclosure should wait until a fix or mitigation is available unless immediate disclosure is necessary to protect users.

## Project-specific security boundary

The public REST coordinator is designed around temporary Bearer-token protection and localhost-only model workers. It is a production-style demo, not a claim of multi-tenant isolation, high availability, or an internet-facing security SLA.
