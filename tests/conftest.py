import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "overflow: mark test as an overflow / edge-case test (run via pexams test-overflow)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--output-dir",
        default=None,
        help="Directory to save test output files. Defaults to a pytest-managed temp directory.",
    )


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory, request):
    custom = request.config.getoption("--output-dir")
    if custom:
        os.makedirs(custom, exist_ok=True)
        return custom
    return str(tmp_path_factory.mktemp("pexams_test"))
