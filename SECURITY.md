# Security Policy

## Supported versions
Only the latest release branch is supported with security patches.

## Reporting a vulnerability
Do not open public issues for vulnerabilities.
Use GitHub Security Advisories or contact maintainers privately.

Include:
- Description
- Steps to reproduce
- Impact assessment
- Proof-of-concept (if safe)
- Proposed mitigation (if available)

## Threats addressed
- Malformed media parsing risks
- Resource exhaustion (very large files)
- Adversarial bypass attempts

## Recommended deployment controls
- File size and duration limits
- Sandboxed decoding
- Rate limiting
- Logging and audit trails
