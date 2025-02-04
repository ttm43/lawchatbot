from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

DATABASE_URL = "postgresql://user:159753@localhost/mike"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


try:
    connection = engine.connect()
    print("Successfully connected to PostgreSQL")
    connection.close()
except Exception as e:
    print(f"Error connecting to PostgreSQL: {e}")


query = "SELECT id, name, question_samples, required_variables_data, answers, tag_ids, required_variables FROM new_question_knowledge;"
kb_df = pd.read_sql(query, engine)

kb_df.to_csv('data/kb.csv', index=False)