import os
import ftplib
import pika
import uuid
import time
import json
from getmac import get_mac_address

# --- KONFIGURASI ---
# RabbitMQ
RABBITMQ_HOST = "rmq230.pptik.id"
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = "terawang"
RABBITMQ_PASSWORD = "Terawang@#2025"
RABBITMQ_VHOST = "/terawang"
RABBITMQ_QUEUE = "terawangDataRow"

# FTP
FTP_HOST = "ftp-sth.pptik.id"
FTP_USER = "terawang"
FTP_PASSWORD = "Terawang@#2025"
FTP_PORT = 2121
FTP_TARGET_FOLDER = "/terawang"

# Folder data lokal
LOCAL_DATA_FOLDER = "data"

def get_formatted_mac():
    """Mendapatkan alamat MAC dan memformatnya dengan tanda hubung."""
    mac = get_mac_address()
    if mac:
        return mac.replace(":", "-").upper()
    # Fallback jika getmac gagal
    print("Tidak bisa mendapatkan MAC address, menggunakan nilai default.")
    return "08-3A-F2-8D-CA-F4"

def main():
    """
    Fungsi utama untuk memproses, mengunggah file ke FTP,
    dan mempublikasikan pesan ke RabbitMQ.
    """
    mac_address = get_formatted_mac()
    print(f"Menggunakan MAC Address: {mac_address}")

    # Loop untuk 8 file JSON
    for i in range(1, 9):
        local_file_path = os.path.join(LOCAL_DATA_FOLDER, f"ketuk{i}.json")

        # Periksa apakah file lokal ada
        if not os.path.exists(local_file_path):
            print(f"❌ File tidak ditemukan: {local_file_path}, melompati...")
            continue

        # 1. Buat metadata dan nama file baru
        guid = str(uuid.uuid4())
        unix_time = int(time.time())
        ketukan_ke = i
        
        # Format nama file baru: mac_guid_unixtime_ketukanke.json
        new_filename = f"{mac_address}_{guid}_{unix_time}_{ketukan_ke}.json"
        
        print(f"\nMemproses file: {os.path.basename(local_file_path)} -> {new_filename}")

        # 2. Kirim file ke FTP
        try:
            print("📤 Mengunggah ke FTP...")
            # Sesuaikan timeout untuk koneksi yang mungkin lambat
            with ftplib.FTP(timeout=30) as ftp:
                ftp.connect(FTP_HOST, FTP_PORT)
                ftp.login(FTP_USER, FTP_PASSWORD)
                ftp.cwd(FTP_TARGET_FOLDER)
                
                with open(local_file_path, 'rb') as f:
                    ftp.storbinary(f'STOR {new_filename}', f)
            
            print(f"✅ Berhasil mengunggah: {new_filename}")

            # 3. Jika berhasil, publish ke RabbitMQ
            try:
                print("📨 Mempublikasikan ke RabbitMQ...")
                credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
                parameters = pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    virtual_host=RABBITMQ_VHOST,
                    credentials=credentials
                )
                connection = pika.BlockingConnection(parameters)
                channel = connection.channel()

                # Pastikan antrian ada
                channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)

                # Buat payload
                payload = {
                    "filename": new_filename,
                    "MAC": mac_address,
                    "GUID": guid,
                    "epoch": unix_time,
                    "ketuk": ketukan_ke
                }
                
                # Kirim pesan
                channel.basic_publish(
                    exchange='',
                    routing_key=RABBITMQ_QUEUE,
                    body=json.dumps(payload, indent=2), # json.dumps untuk konversi ke string
                    properties=pika.BasicProperties(
                        content_type='application/json',
                        delivery_mode=2,  # Buat pesan persistent
                    )
                )

                connection.close()
                print("✅ Pesan berhasil dipublikasikan ke RabbitMQ.")

            except Exception as e:
                print(f"❌ GAGAL mempublikasikan ke RabbitMQ: {e}")

        except ftplib.all_errors as e:
            print(f"❌ GAGAL mengunggah ke FTP: {e}")
        except Exception as e:
            print(f"❌ Terjadi kesalahan tak terduga: {e}")


if __name__ == "__main__":
    # Membuat folder dan file dummy jika belum ada (untuk tujuan pengujian)
    if not os.path.exists(LOCAL_DATA_FOLDER):
        print(f"Membuat folder '{LOCAL_DATA_FOLDER}' untuk pengujian...")
        os.makedirs(LOCAL_DATA_FOLDER)
    for i in range(1, 9):
        filepath = os.path.join(LOCAL_DATA_FOLDER, f"ketuk{i}.json")
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump({"info": f"ini adalah data ketuk ke-{i}"}, f)

    # Jalankan fungsi utama
    main()