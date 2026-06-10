from __future__ import annotations

import os

import pytest

from memlayer import MemoryConfig, MemoryLayer, MissingAPIKeyError
from memlayer.cli import main as cli_main
from memlayer.models import MemoryRecord
from memlayer.storage.sqlite_store import SQLiteMemoryStore

KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI")


@pytest.fixture
def no_api_key(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def test_dotenv_autoload(tmp_path, monkeypatch, no_api_key):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# comment\nGEMINI_API_KEY = \"abc123\"\n\nOTHER=1\n", encoding="utf-8"
    )
    try:
        cfg = MemoryConfig.from_env()
        assert cfg.api_key == "abc123"
    finally:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("OTHER", None)


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    assert MemoryConfig.from_env().api_key == "from-env"


def test_missing_key_error_is_friendly_and_lazy(no_api_key, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # constructing without a key must NOT raise (store-only use is allowed)...
    mem = MemoryLayer(config=MemoryConfig(db_path=":memory:"))
    assert mem.stats()["total"] == 0
    # ...but the first API-needing call raises a helpful error
    with pytest.raises(MissingAPIKeyError) as err:
        mem.search("anything")
    assert "aistudio.google.com" in str(err.value)
    mem.close()


def test_cli_doctor_without_key_fails_helpfully(no_api_key, tmp_path,
                                                monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli_main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "gemini api key" in out
    assert "memlayer init" in out


def test_cli_doctor_with_key_passes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert cli_main(["doctor"]) == 0
    assert "[ok]" in capsys.readouterr().out


def _seed_db(path: str) -> MemoryRecord:
    store = SQLiteMemoryStore(path)
    rec = MemoryRecord(content="hello world note", user_id="default")
    store.add(rec)
    store.close()
    return rec


def test_cli_store_commands_work_without_key(no_api_key, tmp_path,
                                             monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "m.db")
    rec = _seed_db(db)

    assert cli_main(["stats", "--db", db]) == 0
    assert "episodic : 1" in capsys.readouterr().out

    assert cli_main(["list", "--db", db]) == 0
    assert "hello world note" in capsys.readouterr().out

    out_file = str(tmp_path / "dump.json")
    assert cli_main(["export", "--db", db, "-o", out_file]) == 0
    capsys.readouterr()
    assert os.path.exists(out_file)

    assert cli_main(["forget", rec.id, "--db", db]) == 0
    capsys.readouterr()
    assert cli_main(["stats", "--db", db]) == 0
    assert "total    : 0" in capsys.readouterr().out

    # round-trip the export back in without embeddings (keyless path)
    assert cli_main(["import", out_file, "--db", db, "--no-embed"]) == 0
    capsys.readouterr()
    assert cli_main(["clear", "--db", db, "--yes"]) == 0
    assert "Deleted 1" in capsys.readouterr().out


def test_cli_search_without_key_prints_setup_help(no_api_key, tmp_path,
                                                  monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "m.db")
    _seed_db(db)
    assert cli_main(["search", "hello", "--db", db]) == 2
    assert "aistudio.google.com" in capsys.readouterr().err
