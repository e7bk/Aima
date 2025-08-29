#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.database.connection import engine
from src.database.models import Base

def create_tables():
    """Create all database tables"""
    try:
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tables created successfully!")
        
        # Print table info
        print("\nTables created:")
        for table_name, table in Base.metadata.tables.items():
            print(f"  - {table_name}")
            for column in table.columns:
                print(f"    {column.name}: {column.type}")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_tables()