from functools import lru_cache
import sqlalchemy as sa
from rag_decision_engine.config import settings
@lru_cache(maxsize=1)
def get_engine() -> sa.Engine:
    return sa.create_engine(
        settings.postgres_dsn,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
