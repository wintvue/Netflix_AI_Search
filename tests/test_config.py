"""Tests for environment resolution and configuration invariants."""

import importlib

import pytest

import core.config as config


def reload_config(monkeypatch, **env):
    """Reload core.config with a controlled environment."""
    for key in (
        "DB_HOST",
        "DB_PORT",
        "HOST",
        "PORT",
        "RENDER",
        "DYNO",
        "K_SERVICE",
        "WEBSITE_INSTANCE_ID",
        "RERANK_CANDIDATES",
        "MAX_TOP_K",
        "VECTOR_CANDIDATES",
        "BM25_CANDIDATES",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def restore_config():
    yield
    importlib.reload(config)


class TestDatabasePortResolution:
    """The PORT collision: platforms inject PORT for the *web* listener."""

    def test_db_port_is_preferred(self, monkeypatch):
        cfg = reload_config(monkeypatch, DB_PORT=6543, PORT=10000)
        assert cfg.DB_PORT == 6543

    def test_legacy_port_still_works_on_a_dev_machine(self, monkeypatch):
        cfg = reload_config(monkeypatch, PORT=5433)
        assert cfg.DB_PORT == 5433

    def test_legacy_port_is_refused_on_a_managed_platform(self, monkeypatch):
        cfg = reload_config(monkeypatch, PORT=10000, RENDER="true")
        assert cfg.DB_PORT == 5432, "must not use the web port for the database"

    @pytest.mark.parametrize("marker", ["RENDER", "DYNO", "K_SERVICE"])
    def test_every_platform_marker_disables_the_fallback(self, monkeypatch, marker):
        cfg = reload_config(monkeypatch, PORT=10000, **{marker: "1"})
        assert cfg.DB_PORT == 5432

    def test_explicit_db_port_wins_on_a_platform(self, monkeypatch):
        cfg = reload_config(monkeypatch, DB_PORT=6543, PORT=10000, RENDER="true")
        assert cfg.DB_PORT == 6543

    def test_default_when_nothing_is_set(self, monkeypatch):
        cfg = reload_config(monkeypatch)
        assert cfg.DB_PORT == 5432

    def test_non_numeric_port_falls_back(self, monkeypatch):
        cfg = reload_config(monkeypatch, DB_PORT="not-a-port")
        assert cfg.DB_PORT == 5432

    def test_db_host_prefers_the_prefixed_name(self, monkeypatch):
        cfg = reload_config(monkeypatch, DB_HOST="db.internal", HOST="0.0.0.0")
        assert cfg.DB_HOST == "db.internal"

    def test_legacy_host_refused_on_a_platform(self, monkeypatch):
        cfg = reload_config(monkeypatch, HOST="0.0.0.0", RENDER="true")
        assert cfg.DB_HOST is None


class TestMissingDatabaseSettings:
    def test_reports_every_unset_required_setting(self, monkeypatch):
        for key in ("DB_USER", "DB_PASSWORD", "DB_NAME", "DB_HOST"):
            monkeypatch.delenv(key, raising=False)
        cfg = importlib.reload(config)

        assert cfg.missing_db_settings() == [
            "DB_HOST",
            "DB_NAME",
            "DB_PASSWORD",
            "DB_USER",
        ]

    def test_empty_string_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("DB_USER", "")
        cfg = importlib.reload(config)
        assert "DB_USER" in cfg.missing_db_settings()


class TestCandidatePoolInvariants:
    def test_defaults_are_consistent(self, monkeypatch):
        cfg = reload_config(monkeypatch)
        assert cfg.candidate_pool_problems() == []

    def test_detects_the_original_truncation_bug(self, monkeypatch):
        """The shipped config had RERANK_CANDIDATES=50 but served k up to 100."""
        cfg = reload_config(monkeypatch, RERANK_CANDIDATES=50, MAX_TOP_K=100)

        problems = cfg.candidate_pool_problems()

        assert len(problems) == 1
        assert "truncated" in problems[0]

    def test_detects_pools_smaller_than_the_candidate_window(self, monkeypatch):
        cfg = reload_config(
            monkeypatch,
            VECTOR_CANDIDATES=10,
            BM25_CANDIDATES=10,
            RERANK_CANDIDATES=100,
            MAX_TOP_K=100,
        )

        assert any("full pool" in p for p in cfg.candidate_pool_problems())


class TestEnvCoercion:
    def test_int_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("SOME_INT", "abc")
        assert config._env_int("SOME_INT", 7) == 7

    def test_float_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("SOME_FLOAT", "abc")
        assert config._env_float("SOME_FLOAT", 1.5) == 1.5

    def test_empty_string_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("SOME_VALUE", "")
        assert config._env("SOME_VALUE", "fallback") == "fallback"
