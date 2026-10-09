from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import config

# setup engine
# check if sqlite is used to pass connect_args
is_sqlite = config.DATABASE_URL.startswith("sqlite")

if is_sqlite:
    # sqlite needs check_same_thread=False for multithreading in fastapi
    engine = create_engine(
        config.DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    # supabase closes idle connections so we test each one before using it
    # and swap it out after 5 min so a request never gets a dead connection
    engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, pool_recycle=300)

# session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# base model class
Base = declarative_base()


# dependency to get db session in fastapi endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# helper to create all tables
def create_tables():
    # import models here so they get registered on Base metadata
    import models
    Base.metadata.create_all(bind=engine)
