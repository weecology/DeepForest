# Agents Guidelines (Cursor)

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

- Use the project environment managed by `uv`:
  - Sync dev dependencies:
    - `uv sync --dev`
  - Run a specific test or test selection:
    - `uv run pytest -q tests/path/to/test_file.py::test_name`
  - Run the full suite:
    - `uv run pytest`
- If permission prompts appear in Cursor (e.g., network), request and obtain them to complete installs and test runs.

## Testing

- Tests will be run via pytest
- Keep tests short and focused, with a clear contract
- You can find data for testing in src/deepforest/data. Prefer to use existing files with utilities.get_data over generating data at test time.
- Do not use print statements in tests, document failure with assertions
- Use fixtures for repeated code sections, with appropriate scoping
- Run models instead of mocking them. Use existing fixtures if suitable and set config options to be more efficient if necessary (such as fast_dev_run, low numbers of epochs)
- Tests should check for behaviour, not duplicate logic from the library
- Avoid excessive testing for initialization

## Running tests

- From the root folder, `uv run pytest tests`
- Useful flags include `-s` to show output, `-x` to fail on first error and `--ff` to run failed tests first.
