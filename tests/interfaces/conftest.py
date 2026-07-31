"""Reporting support for the batch-wise optimizer benchmark.

A plain ``print`` inside a test is swallowed by pytest's output capture unless the test
fails or ``-s`` is passed. The measured timings are worth seeing on an ordinary run, so
they are collected here and written out through the terminal reporter in the summary
section at the end of the run instead.
"""

import pytest


_REPORT_LINES = []


@pytest.fixture
def benchmark_report():
    """Return a callable that queues a line for the end of run summary."""
    return _REPORT_LINES.append


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _REPORT_LINES:
        return
    terminalreporter.write_sep("=", "batch-wise vs sequential relaxation")
    for line in _REPORT_LINES:
        terminalreporter.write_line(line)
