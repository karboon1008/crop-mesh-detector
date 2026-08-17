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


def test_record_transfer_then_read_transfers_round_trips(tmp_path):
    db_path = tmp_path / "node_0.db"
    sqlite_store.record_transfer(db_path, 0, "node_1", "node_0", 1234, "2026-08-17T00:00:00Z")

    rows = sqlite_store.read_transfers(db_path)
    assert len(rows) == 1
    assert rows[0]["round_idx"] == 0
    assert rows[0]["from_node"] == "node_1"
    assert rows[0]["to_node"] == "node_0"
    assert rows[0]["size_bytes"] == 1234
    assert rows[0]["fetched_at"] == "2026-08-17T00:00:00Z"


def test_read_transfers_keeps_insertion_order_across_multiple_rounds(tmp_path):
    db_path = tmp_path / "node_0.db"
    sqlite_store.record_transfer(db_path, 0, "node_1", "node_0", 100, "t0")
    sqlite_store.record_transfer(db_path, 0, "node_2", "node_0", 200, "t1")
    sqlite_store.record_transfer(db_path, 1, "node_1", "node_0", 150, "t2")

    rows = sqlite_store.read_transfers(db_path)
    assert [r["size_bytes"] for r in rows] == [100, 200, 150]


def test_read_transfers_on_missing_db_returns_empty_list(tmp_path):
    assert sqlite_store.read_transfers(tmp_path / "does_not_exist.db") == []


def test_read_transfers_on_existing_db_without_the_table_returns_empty_list(tmp_path):
    # Simulates a read-only-mounted db file (the dashboard's view) that
    # predates the knowledge_transfers table -- must never attempt a schema
    # write just to discover there are no rows.
    import sqlite3

    db_path = tmp_path / "node_0.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(sqlite_store.SCHEMA_SQL)  # only round_metrics, no knowledge_transfers
    conn.commit()
    conn.close()
    assert sqlite_store.read_transfers(db_path) == []


def test_read_all_on_existing_db_without_the_table_returns_empty_list(tmp_path):
    import sqlite3

    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))
    conn.close()
    assert sqlite_store.read_all(db_path) == []


def test_record_transfer_does_not_touch_round_metrics_table(tmp_path):
    # Transfers are recorded from the fetching side, independent of the
    # round_metrics rows the coordinator/node write via upsert_row.
    db_path = tmp_path / "node_0.db"
    sqlite_store.record_transfer(db_path, 0, "node_1", "node_0", 100, "t0")
    assert sqlite_store.read_all(db_path) == []
