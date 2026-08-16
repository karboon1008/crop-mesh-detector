from __future__ import annotations

from src.energy import sqlite_store


def test_upsert_row_creates_row_with_given_columns(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite_store.upsert_row(db_path, "node_0", 0, "2026-08-16T00:00:00Z", energy_kwh=0.001, duration_s=1.5)

    rows = sqlite_store.read_all(db_path)
    assert len(rows) == 1
    assert rows[0]["node_id"] == "node_0"
    assert rows[0]["round_idx"] == 0
    assert rows[0]["energy_kwh"] == 0.001
    assert rows[0]["duration_s"] == 1.5
    assert rows[0]["crop_accuracy"] is None


def test_upsert_row_merges_separate_messages_into_one_row(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite_store.upsert_row(
        db_path, "node_0", 0, "2026-08-16T00:00:00Z",
        energy_kwh=0.001, duration_s=1.5, energy_method="proxy_wall_power",
    )
    sqlite_store.upsert_row(
        db_path, "node_0", 0, "2026-08-16T00:00:01Z",
        crop_accuracy=0.8, disease_accuracy=0.7,
    )

    rows = sqlite_store.read_all(db_path)
    assert len(rows) == 1
    row = rows[0]
    # both writes' columns survive -- the second upsert must not null out the first's
    assert row["energy_kwh"] == 0.001
    assert row["energy_method"] == "proxy_wall_power"
    assert row["crop_accuracy"] == 0.8
    assert row["disease_accuracy"] == 0.7
    assert row["recorded_at"] == "2026-08-16T00:00:01Z"  # most recent write wins


def test_upsert_row_keeps_separate_rows_per_node_and_round(tmp_path):
    db_path = tmp_path / "test.db"
    sqlite_store.upsert_row(db_path, "node_0", 0, "t0", crop_accuracy=0.5)
    sqlite_store.upsert_row(db_path, "node_1", 0, "t0", crop_accuracy=0.6)
    sqlite_store.upsert_row(db_path, "node_0", 1, "t1", crop_accuracy=0.55)

    rows = sqlite_store.read_all(db_path)
    assert len(rows) == 3


def test_read_all_on_missing_db_returns_empty_list(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert sqlite_store.read_all(db_path) == []
