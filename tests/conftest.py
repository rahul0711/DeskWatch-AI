"""Keep the test suite off the real database and the real login system.

app.config and app.activity.config call load_dotenv(override=True) when they
are first imported, which copies DATABASE_URL (the shared MySQL server) and
JWT_SECRET from .env into the environment. Import them first so that has
already happened, then remove those variables for every test: tests fall back
to their own temporary SQLite files and the legacy demo-token login.
"""
import os

import pytest

import app.activity.config  # noqa: F401  -- runs load_dotenv before we clear
import app.config  # noqa: F401

_PRODUCTION_ONLY = ("DATABASE_URL", "JWT_SECRET", "JWT_ISSUER", "JWT_AUDIENCE")

for _name in _PRODUCTION_ONLY:
    os.environ.pop(_name, None)


@pytest.fixture(autouse=True)
def _no_production_settings(monkeypatch):
    for name in _PRODUCTION_ONLY:
        monkeypatch.delenv(name, raising=False)
