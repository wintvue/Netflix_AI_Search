import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DB_URL = os.environ["DATABASE_URL"]  # e.g. postgres://...
with psycopg.connect(DB_URL) as conn:
        conn.execute("SET statement_timeout = 0;")