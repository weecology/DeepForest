## Testing

- When proposing code changes that require new tests:
  - First add the minimal failing test(s) that reproduce the issue or specify the expected new behavior.
  - Run those specific tests and show their failing output (red) before suggesting or implementing a fix.
  - After implementing the fix, re-run the same tests and show they pass (green).
- Keep tests short and focused, with a clear contract.
- Prefer using existing data via `deepforest.get_data(...)` over generating new data at runtime.
- Do not use print statements in tests; document failure with assertions.
- Use fixtures for repeated setup; keep scope appropriate.
- Test behavior, not implementation details.

## Running tests

- From the root folder, `uv run pytest tests`
- Useful flags include `-s` to show output, `-x` to fail on first error and `--ff` to run failed tests first.
