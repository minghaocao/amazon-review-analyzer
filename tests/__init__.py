"""Amazon Review Analyzer tests with shared offline UI configuration.

Streamlit AppTest executes the dashboard as a separate script.  Supplying a
deterministic UI health profile keeps those rendering tests fully offline;
``test_environment_health`` exercises the real checker through injected fake
probes instead.
"""

import os


os.environ.setdefault("ARA_UI_HEALTH_TEST_PROFILE", "ready")
