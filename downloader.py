import json
import pika
import ftplib
import io
import time
import os
import uuid
from dotenv import load_dotenv

# Muat variabel dari file .env
load_dotenv()

# --- KONFIGURASI DARI .ENV ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT"))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST")
RABBITMQ_INPUT_QUEUE = os.getenv("RABBITMQ_INPUT_QUEUE")
RABBITMQ_FOLDER_QUEUE = "terawang-folder"

FTP_HOST = os.getenv("FTP_HOST")
FTP_PORT = int(os.getenv("FTP_PORT"))
FTP_USER = os.getenv("FTP_USER")
FTP_PASSWORD = os.getenv("FTP_PASSWORD")
FTP_SOURCE_FOLDER = os.getenv("FTP_SOURCE_FOLDER")

# Variabel global untuk melacak proses unduhan per GUID
download_tracker = {}

# ====================================================================
# FUNGSI HELPER
# ====================================================================

def download_json_from_ftp(filename):
    """Mengunduh file JSON dari server FTP dan mengembalikannya sebagai objek Python."""
    print(f"  ⬇️   Mencoba mengunduh '{filename}' dari FTP...")
    try:
        with ftplib.FTP(timeout=30) as ftp:
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASSWORD)
            ftp.cwd(FTP_SOURCE_FOLDER)

            mem_file = io.BytesIO()
            ftp.retrbinary(f'RETR {filename}', mem_file.write)
            mem_file.seek(0)

            data = json.load(io.TextIOWrapper(mem_file, encoding='utf-8'))
            print(f"  ✅  File '{filename}' berhasil diunduh dan diparsing.")
            return data
    except ftplib.all_errors as e:
        print(f"  ❌  GAGAL mengunduh dari FTP: {e}")
        return None

def save_json_locally(folder_path, filename, data):
    """Menyimpan data (objek Python) sebagai file JSON di folder lokal."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"  📁  Folder '{folder_path}' berhasil dibuat.")

        local_filepath = os.path.join(folder_path, filename)
        with open(local_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"  💾  File '{filename}' berhasil disimpan di '{folder_path}'.")
        return True
    except Exception as e:
        print(f"  ❌  GAGAL menyimpan file secara lokal: {e}")
        return False

def publish_folder_to_rmq(channel, folder_name):
    """Mempublikasikan nama folder sebagai plain text ke antrian RabbitMQ."""
    try:
        channel.basic_publish(
            exchange='',
            routing_key=RABBITMQ_FOLDER_QUEUE,
            body=folder_name.encode('utf-8'),
            properties=pika.BasicProperties(
                content_type='text/plain',
                delivery_mode=2,  # make message persistent
            )
        )
        print(f"  📨  Nama folder '{folder_name}' BERHASIL dipublikasikan ke antrian '{RABBITMQ_FOLDER_QUEUE}'.")
    except Exception as e:
        print(f"  ❌  GAGAL mempublikasikan ke RabbitMQ: {e}")

def get_guid_from_data(data_list):
    """Mengekstrak 'guidteensy' dari data yang diunduh."""
    if isinstance(data_list, list):
        for item in data_list:
            if isinstance(item, dict) and 'guidteensy' in item:
                return item['guidteensy']
    return None

# ====================================================================
# LOGIKA UTAMA (CALLBACK) - VERSI FINAL
# ====================================================================

def callback(ch, method, properties, body):
    global download_tracker
    print(f"\n[+] Pesan baru diterima dari '{RABBITMQ_INPUT_QUEUE}'")
    try:
        message = json.loads(body)
        filename = message.get("filename") # Contoh: ..._8.json

        if not filename:
            print("  ❌  Pesan tidak valid (tidak ada 'filename'). Diabaikan.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        downloaded_data = download_json_from_ftp(filename)

        if not downloaded_data:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        guid_survey = get_guid_from_data(downloaded_data)
        if not guid_survey:
            print(f"  ❌  'guidteensy' tidak ditemukan dalam file {filename}. Diabaikan.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        if guid_survey not in download_tracker:
            folder_name = str(uuid.uuid4())
            download_tracker[guid_survey] = {
                "folder_name": folder_name,
                "files_downloaded": 0
            }
            print(f"  🆕  Memulai sesi unduhan baru untuk GUID {guid_survey} di folder '{folder_name}'")

        session_folder = download_tracker[guid_survey]["folder_name"]
        
        # --- PERUBAHAN LOGIKA PENAMAAN FILE DIMULAI DI SINI ---
        try:
            # 1. Pisahkan nama file dari ekstensinya (".json")
            # Contoh: "20-79..._8.json" -> "20-79..._8"
            base_name = filename.rsplit('.', 1)[0]
            
            # 2. Pisahkan berdasarkan underscore "_" dan ambil bagian terakhir
            # Contoh: "20-79..._8" -> "8"
            file_number = base_name.split('_')[-1]

            # 3. Buat nama file baru berdasarkan nomor yang diekstrak
            # Contoh: "ketukan8.json"
            new_filename = f"ketukan{file_number}.json"
            
        except IndexError:
            # Jika nama file tidak sesuai format (misal: "test.json" tanpa "_")
            print(f"  ⚠️  Format nama file '{filename}' tidak standar. Menggunakan nama asli.")
            new_filename = filename # Fallback, gunakan nama asli
        # --- AKHIR PERUBAHAN ---

        print(f"  -> Mengubah nama file '{filename}' menjadi '{new_filename}'")

        if save_json_locally(session_folder, new_filename, downloaded_data):
            # Counter tetap digunakan untuk melacak kapan 8 file selesai diunduh
            download_tracker[guid_survey]["files_downloaded"] += 1
            count = download_tracker[guid_survey]['files_downloaded']
            print(f"  👍  Progres GUID {guid_survey}: ({count}/8) file terkumpul.")

        if download_tracker[guid_survey]["files_downloaded"] == 8:
            print(f"\n✨ Semua 8 file untuk GUID {guid_survey} telah diunduh!")

            publish_folder_to_rmq(ch, session_folder)

            del download_tracker[guid_survey]
            print(f"  ✅  Sesi untuk GUID {guid_survey} selesai dan dihapus dari tracker.")

    except json.JSONDecodeError:
        print("  ❌  Pesan bukan JSON yang valid. Diabaikan.")
    except Exception as e:
        print(f"  ❌  Terjadi kesalahan tak terduga saat pemrosesan: {e}")

    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST, port=RABBITMQ_PORT, virtual_host=RABBITMQ_VHOST,
        credentials=credentials, heartbeat=600, blocked_connection_timeout=300
    )

    print("Menghubungkan ke RabbitMQ...")
    while True:
        try:
            with pika.BlockingConnection(parameters) as connection:
                channel = connection.channel()
                channel.queue_declare(queue=RABBITMQ_INPUT_QUEUE, durable=True)
                channel.queue_declare(queue=RABBITMQ_FOLDER_QUEUE, durable=True)

                channel.basic_qos(prefetch_count=1)
                channel.basic_consume(queue=RABBITMQ_INPUT_QUEUE, on_message_callback=callback)

                print(f"✅ Terhubung! Menunggu pesan di antrian '{RABBITMQ_INPUT_QUEUE}'...")
                channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Koneksi gagal: {e}. Mencoba lagi dalam 5 detik...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nProgram dihentikan oleh pengguna.")
            break
        except Exception as e:
            print(f"Kesalahan tidak terduga: {e}. Memulai ulang...")
            time.sleep(5)

if __name__ == '__main__':
    main()