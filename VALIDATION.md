# Validation

Checked on 2026-09-18 using the staged public source on Windows.

- Offline pytest suite: 235 passed in 136.24 seconds.
- The CLI coverage-report tests now replace the optional AkShare provider with
  their existing fake provider. Production code is unchanged by that adjustment.
- Reused the bundled scientific Python packages and an existing local pytest
  environment. A clean install of all dependencies was not exercised.
- No live market data, broker, account or model/API request was requested.
- Synthetic tests and smoke outputs are not trading performance evidence.

Market caches, personal research outputs and the original local Git history are
excluded from this publication snapshot.
