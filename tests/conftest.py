"""Shared pytest configuration — adds project root to sys.path."""
import sys
import os
from unittest.mock import MagicMock

# Ensure project root is on the path before any test imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Permanently stub out UI/dashboard modules that are never tested directly.
# These stubs must be set before any test module is imported so that
# application.py can be imported without a running Dash server or display.
# Crucially, these are set ONCE here so there is no context-manager
# save/restore cycle that could corrupt C-extension (numpy) module state.
_UI_STUBS = [
    "dash",
    "dash.dcc",
    "dash.html",
    "dash.dependencies",
    "dash_bootstrap_components",
    "dash_chat",
    "dash_chat_components",
    "make_figures",
]
for _mod in _UI_STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
