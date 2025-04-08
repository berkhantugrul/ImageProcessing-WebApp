import sqlite3
from datetime import datetime

DB_NAME = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih TEXT NOT NULL,
            saat TEXT NOT NULL,
            dosya_adi TEXT NOT NULL,
            metot TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_process(dosya_adi, metot):
    now = datetime.now()
    tarih = now.strftime('%Y-%m-%d')
    saat = now.strftime('%H:%M:%S')
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO processed_images (tarih, saat, dosya_adi, metot) VALUES (?, ?, ?, ?)",
        (tarih, saat, dosya_adi, metot)
    )
    conn.commit()
    conn.close()

def get_statistics():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM processed_images")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT dosya_adi) FROM processed_images")
    unique = cursor.fetchone()[0]
    cursor.execute("SELECT metot, COUNT(*) FROM processed_images GROUP BY metot")
    per_technique = cursor.fetchall()
    conn.close()
    return total, unique, per_technique

def get_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT tarih, saat, dosya_adi, metot FROM processed_images ORDER BY id DESC")
    logs = cursor.fetchall()
    conn.close()
    return logs
