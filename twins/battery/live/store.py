"""SQLite storage for the live twin: measurements, twin state, decisions."""
import os
import sqlite3
from pathlib import Path

# Override with DT_DB=/some/local/path.db when the project folder is on a network/VM mount
# that does not support SQLite file locks.
DB = Path(os.environ.get("DT_DB", Path(__file__).resolve().parents[1] / "results" / "live.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (cell TEXT, cycle INTEGER, q REAL, cap_ah REAL, t REAL);
CREATE TABLE IF NOT EXISTS state (cell TEXT, cycle INTEGER, log10_a REAL, b REAL, log10_a_std REAL,
    b_std REAL, eol_pred REAL, rul_lo REAL, rul_hi REAL, t REAL);
CREATE TABLE IF NOT EXISTS decisions (cell TEXT, cycle_decided INTEGER, x_twin INTEGER,
    horizon INTEGER, t REAL);
CREATE TABLE IF NOT EXISTS meta (cell TEXT PRIMARY KEY, q0 REAL, k_fpt INTEGER, last_measured INTEGER);
"""


def connect(reset=False):
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, check_same_thread=False)
    if reset:  # clear tables instead of deleting the file
        con.executescript("DROP TABLE IF EXISTS measurements; DROP TABLE IF EXISTS state;"
                          "DROP TABLE IF EXISTS decisions; DROP TABLE IF EXISTS meta;")
    con.executescript(SCHEMA)
    return con
