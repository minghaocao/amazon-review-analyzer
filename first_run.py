"""Safe persistence for the single-screen first-run experience.

The state is deliberately tiny and contains no user, review, account, or
filesystem information.  UI code may supply a different path in tests, while
normal application use writes below the project's existing runtime output
area.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FIRST_RUN_STATE_PATH = (
    PROJECT_ROOT / "outputs" / "app_state" / "first_run.json"
)


@dataclass(frozen=True)
class FirstRunState:
    """Validated first-run state exposed to presentation-layer callers."""

    first_run_completed: bool = False
    app_version: str = ""


def load_first_run_state(
    state_path: str | Path = DEFAULT_FIRST_RUN_STATE_PATH,
) -> FirstRunState:
    """Load validated state, returning a safe incomplete state on any error.

    Corrupt, unreadable, or schema-invalid state must never prevent the
    dashboard from starting.  Unknown fields are ignored and never returned.
    """

    destination = Path(state_path)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return FirstRunState()

    if not isinstance(payload, dict):
        return FirstRunState()

    completed = payload.get("first_run_completed")
    version = payload.get("app_version")
    if not isinstance(completed, bool):
        return FirstRunState()
    if not isinstance(version, str):
        return FirstRunState()

    normalized_version = version.strip()
    if completed and not normalized_version:
        return FirstRunState()
    return FirstRunState(
        first_run_completed=completed,
        app_version=normalized_version,
    )


def is_first_run_completed(
    state_path: str | Path = DEFAULT_FIRST_RUN_STATE_PATH,
) -> bool:
    """Return whether onboarding was explicitly completed or skipped."""

    return load_first_run_state(state_path).first_run_completed


def mark_first_run_completed(
    app_version: str,
    state_path: str | Path = DEFAULT_FIRST_RUN_STATE_PATH,
) -> FirstRunState:
    """Persist completion atomically and return the stored state.

    The temporary file lives beside the destination so ``os.replace`` remains
    an atomic filesystem operation.  An existing good state is preserved if a
    write or replace fails.
    """

    if not isinstance(app_version, str) or not app_version.strip():
        raise ValueError("app_version must be a non-empty string")

    state = FirstRunState(
        first_run_completed=True,
        app_version=app_version.strip(),
    )
    destination = Path(state_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                asdict(state),
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass

    return state


def reset_first_run_state_for_tests(state_path: str | Path) -> None:
    """Remove a caller-supplied state file for isolated automated tests only.

    Requiring an explicit path prevents production code from accidentally
    resetting the default first-run experience.
    """

    try:
        Path(state_path).unlink()
    except FileNotFoundError:
        return

