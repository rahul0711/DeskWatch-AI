"""One-time copy of the attendance data from SQLite (data/attendance.db) into
MySQL: users, face_embeddings (the AdaFace fingerprints, byte-for-byte) and
attendance_events. Row ids are kept, so user_id links and the image files in
data/enrollment/ and data/attendance_crops/ stay valid.

Refuses to run if any target table already has rows -- it never merges or
overwrites. The SQLite file is only read, never changed.

Usage:
    source .venv/bin/activate
    python scripts/migrate_sqlite_to_mysql.py --mysql-url "mysql+pymysql://attendance:PASS@127.0.0.1:3306/attendance"
Then set the same URL as DATABASE_URL in .env and restart the backend.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, func, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attendance.config import ROOT  # noqa: E402
from app.attendance.db import Base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sqlite-path", default=str(ROOT / "data" / "attendance.db"))
    parser.add_argument("--mysql-url", required=True)
    args = parser.parse_args()

    if not Path(args.sqlite_path).exists():
        print(f"SQLite file not found: {args.sqlite_path}", file=sys.stderr)
        return 1

    src = create_engine(f"sqlite:///{args.sqlite_path}")
    dst = create_engine(args.mysql_url)
    Base.metadata.create_all(dst)

    tables = Base.metadata.sorted_tables  # parents first: users before its FKs
    with dst.connect() as conn:
        for table in tables:
            n = conn.execute(select(func.count()).select_from(table)).scalar_one()
            if n:
                print(f"Target table '{table.name}' already has {n} rows -- aborting, nothing copied.", file=sys.stderr)
                return 1

    with src.connect() as s, dst.begin() as d:
        for table in tables:
            rows = [dict(r) for r in s.execute(select(table)).mappings()]
            if rows:
                d.execute(table.insert(), rows)
            print(f"{table.name}: copied {len(rows)} rows")

    with src.connect() as s, dst.connect() as d:
        for table in tables:
            a = s.execute(select(func.count()).select_from(table)).scalar_one()
            b = d.execute(select(func.count()).select_from(table)).scalar_one()
            status = "OK" if a == b else "MISMATCH"
            print(f"verify {table.name}: sqlite={a} mysql={b} {status}")
            if a != b:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
