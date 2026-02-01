# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability:

1.  **Do NOT open a GitHub Issue.**
2.  Email `security@financial-swarm.local` immediately.
3.  We will acknowledge within 24 hours.

### Input Sanitization
This project uses `src.utils.validation.sanitize_input` to scrub CLI arguments.

### Dependencies
We use `requirements.txt` with pinned versions to prevent Supply Chain Attacks.
