import ftplib
import os

# --- Konfigurasi FTP ---
FTP_HOST = "ftp-sth.pptik.id"
FTP_PORT = 2121
FTP_USER = "terawang"
FTP_PASSWORD = "Terawang@#2025"

# --- Konfigurasi File & Folder ---
LOCAL_FILE_NAME = "d4714642-7146-4b2c-bf19-853d412d448c.json"
REMOTE_FOLDER = "/result"
REMOTE_FILE_NAME = "d4714642-7146-4b2c-bf19-853d412d448c.json"

# --- Pastikan file lokal ada (opsional, untuk pengujian) ---
if not os.path.exists(LOCAL_FILE_NAME):
    print(f"Membuat file dummy '{LOCAL_FILE_NAME}' untuk pengujian.")
    with open(LOCAL_FILE_NAME, 'w') as f:
        f.write('{"status": "test", "message": "file ini dibuat untuk pengujian upload"}')

# --- Proses Upload ---
try:
    # 1. Menghubungkan ke server FTP
    print(f"Menghubungkan ke {FTP_HOST}:{FTP_PORT}...")
    ftp = ftplib.FTP()
    ftp.connect(FTP_HOST, FTP_PORT)
    
    # 2. Login dengan user dan password
    print(f"Login sebagai {FTP_USER}...")
    ftp.login(FTP_USER, FTP_PASSWORD)
    
    # 3. Berpindah ke folder tujuan di server
    print(f"Berpindah ke direktori '{REMOTE_FOLDER}'...")
    ftp.cwd(REMOTE_FOLDER)
    
    # 4. Membuka file lokal dalam mode binary read ('rb')
    with open(LOCAL_FILE_NAME, 'rb') as file:
        # 5. Mengirim file ke server
        print(f"Mengunggah {LOCAL_FILE_NAME} ke {REMOTE_FOLDER}...")
        ftp.storbinary(f'STOR {REMOTE_FILE_NAME}', file)
    
    print("✅ Upload berhasil!")

except ftplib.all_errors as e:
    print(f"❌ Terjadi error pada FTP: {e}")
    print("Pastikan folder '/result' sudah ada di server FTP.")

finally:
    # 6. Menutup koneksi
    if 'ftp' in locals() and ftp.sock is not None:
        ftp.quit()
        print("Koneksi ditutup.")