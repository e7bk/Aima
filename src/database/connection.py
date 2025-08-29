from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.utils.config import Config

config = Config()
engine = create_engine(config.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
