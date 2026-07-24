# Security policy

## Supported version

Security fixes are applied to the latest revision of the `main` branch.

## Reporting a vulnerability

Do not disclose credentials or exploitable details in a public issue. Use the
repository's private vulnerability-reporting option under the GitHub
**Security** tab when it is available. Otherwise, contact the maintainer through
the GitHub profile linked from the repository and arrange a private channel
before sharing details.

Include:

- the affected file, module, or dependency;
- steps to reproduce the issue;
- the likely impact; and
- any suggested mitigation.

## Credential handling

The local simulator and automated test suite do not need cloud credentials.
Optional IBM Quantum credentials must be provided at runtime through an
environment variable or an approved local credential store.

Never place a token in source code, a notebook, test output, an experiment
artifact, or a GitHub issue. If a token reaches a commit:

1. revoke or rotate it immediately;
2. remove it from the current tree;
3. audit account activity; and
4. treat every copy in Git history, forks, and caches as compromised.

Rewriting Git history does not make rotation optional.
