# CHANGELOG


## v0.2.0-rc.1 (2025-05-04)

### Testing

- **database**: Skip database-dependent tests in ci environment
  ([`50ed567`](https://github.com/SarthakMishra/codemap/commit/50ed567554693bc4a5b778a4a375b7588052ab86))

Added environment variable SKIP_DB_TESTS to skip database-dependent tests in CI environments without
  PostgreSQL. Updated test_client.py and test_models.py to use the skip_db_tests marker from
  conftest.py.


## v0.1.0 (2025-05-02)


## v0.1.0-rc.1 (2025-05-02)

- Initial Release
