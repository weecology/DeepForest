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
