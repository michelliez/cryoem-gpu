import pytest

_REPORT_KEY = pytest.StashKey[list]()


def pytest_configure(config):
    config.stash[_REPORT_KEY] = []


@pytest.fixture
def agreement_report(request):
    """Append a line to the agreement report printed at the end of the run, pass or fail."""
    return request.config.stash[_REPORT_KEY].append


def pytest_terminal_summary(terminalreporter, config):
    lines = config.stash[_REPORT_KEY]
    if lines:
        terminalreporter.section("agreement report")
        for line in lines:
            terminalreporter.write_line(line)
