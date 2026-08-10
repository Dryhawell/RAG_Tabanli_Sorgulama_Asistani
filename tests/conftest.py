import os
import sys

import pytest

# Ensure project root is on sys.path for imports like `import rag`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: gerçek model indirmesi gerektiren uzun testler (RUN_SLOW_EVAL=1)",
    )


def pytest_collection_modifyitems(config, items):
    """Varsayılan olarak slow testleri atla; RUN_SLOW_EVAL=1 ile açılır."""
    run_slow = os.getenv("RUN_SLOW_EVAL", "").lower() in {"1", "true", "yes"}
    if run_slow:
        return
    skip_slow = pytest.mark.skip(reason="slow eval kapalı; RUN_SLOW_EVAL=1 ile çalıştırın")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
