# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Shared pytest fixtures and CLI options for the vllm-qaic test suite."""

import ast

import pytest


def pytest_addoption(parser):
    """Register the --test-device-group option.

    Mirrors the convention documented in CLAUDE.md, e.g.::

        pytest -s tests/test_qaic/spec_decode/e2e/... --test-device-group '[4,5,6,7]'
    """
    parser.addoption(
        "--test-device-group",
        action="store",
        default="[0]",
        help=(
            "QAIC device group (QIDs) as a Python list literal, "
            "e.g. '[0]' or '[4,5,6,7]'. Passed to the LLM via "
            "additional_config['device_group']."
        ),
    )


@pytest.fixture(scope="session")
def device_group(request) -> list[int]:
    """Parse --test-device-group into a list of QIDs."""
    raw = request.config.getoption("--test-device-group")
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise pytest.UsageError(
            f"--test-device-group must be a Python list literal, got {raw!r}"
        ) from exc
    if not isinstance(parsed, (list, tuple)):
        raise pytest.UsageError(
            f"--test-device-group must parse to a list/tuple, got {type(parsed)}"
        )
    return list(parsed)
