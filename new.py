import sqlite3
import logging

def setup_database():
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()

        # Drop existing table if needed
        cursor.execute("DROP TABLE IF EXISTS attendance")

        # Create new table with snapshot tracking
        cursor.execute("""
        CREATE TABLE attendance (
            name TEXT,
            time TEXT,
            date DATE,
            snapshots INTEGER DEFAULT 0,
            status TEXT,
            UNIQUE(name, date)
        )
        """)

        conn.commit()
        logging.info("Database setup completed successfully")
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_database()