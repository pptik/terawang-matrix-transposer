#!/usr/bin/env python
# coding: utf-8

import numpy as np
import codecs, json
import time
import os
import pika
import uuid # Ditambahkan untuk menghasilkan UUID
from ftplib import FTP
from collections import defaultdict
from datetime import datetime
import shutil
from dotenv import load_dotenv

# Muat variabel dari file .env
load_dotenv()

# =============================================================================
# KONFIGURASI APLIKASI (DIAMBIL DARI .env)
# =============================================================================
# Konfigurasi Folder Lokal
LOCAL_RESULT_DIR = "hasil_proses_lokal"
LOCAL_TEMP_DIR = "temp_data" # Folder untuk mengunduh file & menyimpan hasil sementara

# Akses FTP
FTP_HOST = os.getenv("FTP_HOST", "ftp-sth.pptik.id")
FTP_PORT = int(os.getenv("FTP_PORT", 2121))
FTP_USER = os.getenv("FTP_USER", "terawang")
FTP_PASSWORD = os.getenv("FTP_PASSWORD", "Terawang@#2025")
FTP_SOURCE_FOLDER = os.getenv("FTP_SOURCE_FOLDER", "/terawang") # Folder sumber di FTP
FTP_FOLDER_HASIL = os.getenv("FTP_FOLDER_HASIL", "/result")
FTP_FOLDER_DATA_ROW = os.getenv("FTP_FOLDER_DATA_ROW", "/data_row")

# Akses RabbitMQ
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rmq230.pptik.id")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME", "terawang")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "Terawang@#2025")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/terawang")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE_RESULT", "result_queue")
RABBITMQ_GRAPH_QUEUE = os.getenv("RABBITMQ_GRAPH_QUEUE", "graph_row")

# Konfigurasi Pemrosesan
DIAMETER = 0.3
PROCESSING_INTERVAL_SECONDS = 10
GROUPING_TIME_WINDOW_MINUTES = 15


# =============================================================================
# FUNGSI PEMROSESAN SINYAL (TIDAK DIUBAH)
# =============================================================================
def load_sigarray_from_json(filename):
    """Memuat data sensor dan timestamp dari satu file JSON."""
    all_sensor_data = {}
    timestamp_data = None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            json_content = json.load(f)
    except Exception as e:
        print(f"Error reading/parsing JSON {filename}: {e}")
        return None

    for item in json_content:
        if 'timestamp' in item:
            timestamp_data = item['timestamp']
        for i in range(1, 9):
            key = f'value{i}'
            if key in item:
                all_sensor_data[key] = item[key]
                
    try:
        if timestamp_data is None or len(all_sensor_data) != 8:
            raise ValueError(f"Error: Data tidak lengkap di file {filename}")
    except ValueError as ve:
        print(ve)
        return None
            
    try:
        sensor_arrays = [all_sensor_data[f'value{i}'] for i in range(1, 9)]
        arrays_to_stack = sensor_arrays + [timestamp_data]
        min_len = min(len(arr) for arr in arrays_to_stack)
        sigarray = np.column_stack([np.array(arr[:min_len]) for arr in arrays_to_stack])
        return sigarray
    except Exception as e:
        print(f"Error saat menyusun numpy array: {e}")
        return None

def gcc(sig, refsig, fs=1000000, CCType="PHAT", max_tau=None):
    """Menghitung Generalized Cross-Correlation."""
    n = len(sig)
    sig = sig - np.mean(sig)
    refsig = refsig - np.mean(refsig)

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    
    if CCType.upper() == "PHAT":
        WEIGHT = 1 / (np.abs(R) + 1e-10)
    else:
        WEIGHT = 1.0
    
    Integ = R * WEIGHT
    cc = np.fft.irfft(Integ, n=n)
    cc = np.fft.fftshift(cc)

    if max_tau is not None:
        max_shift = int(fs * max_tau)
    else:
        max_shift = n // 2

    center_index = n // 2
    search_range = slice(center_index - max_shift, center_index + max_shift)
    
    shift = np.argmax(np.abs(cc[search_range])) - max_shift
    tau = shift / float(fs)
    
    return np.abs(tau), cc, None

def onetap(sigarray, which, diameter):
    """Menghitung kecepatan dari satu set data ketukan."""
    sensors = [(sigarray[:, i] for i in range(8))] # Cuts through sigarray (which is a 2d Array) 
    # and returms them to sensors per column 
    radius = diameter / 2
    distances = {
        (1,2): radius * 0.765, (1,3): radius * 1.414, (1,4): radius * 1.847, (1,5): diameter,
        (1,6): radius * 1.847, (1,7): radius * 1.414, (1,8): radius * 0.765,
        (2,3): radius * 0.765, (2,4): radius * 1.414, (2,5): radius * 1.847, (2,6): diameter,
        (2,7): radius * 1.847, (2,8): radius * 1.414,
        (3,4): radius * 0.765, (3,5): radius * 1.414, (3,6): radius * 1.847, (3,7): diameter,
        (3,8): radius * 1.847,
        (4,5): radius * 0.765, (4,6): radius * 1.414, (4,7): radius * 1.847, (4,8): diameter,
        (5,6): radius * 0.765, (5,7): radius * 1.414, (5,8): radius * 1.847,
        (6,7): radius * 0.765, (6,8): radius * 1.414,
        (7,8): radius * 0.765
    }
    
    refsig = sensors[which - 1]
    velocities = np.zeros(8)
    
    for i in range(8):
        if i == (which - 1): continue
        sig = sensors[i]
        pair = tuple(sorted((which, i + 1)))
        if len(pair) != 2 or not all(isinstance(x, int) for x in pair):
            raise ValueError("Pair must be a tuple of two integers")
        if pair not in distances:
            raise ValueError(f"Pair {pair} not found in distances dictionary")
        dist = distances.get(pair)
        if dist is None: continue
        tof = gcc(refsig=refsig, sig=sig)[0]
        velocities[i] = dist / tof if tof > 1e-9 else np.inf
    return velocities.astype(np.float32)

# =============================================================================
# FUNGSI UTAMA UNTUK ORKESTRASI PEMROSESAN
# =============================================================================
def process_batch(file_paths, diameter):
    """Memproses satu batch (8 file) dan mengembalikan matriks kecepatan."""
    all_velocity_rows = []
    sorted_paths = sorted(file_paths)
    for i, filepath in enumerate(sorted_paths, start=1):
        sigarray = load_sigarray_from_json(filepath)
        if sigarray is not None:
            velocity_row = onetap(sigarray, which=i, diameter=diameter)
            all_velocity_rows.append(velocity_row)
        else:
            print(f"Gagal memproses {filepath}, baris akan diisi nol.")
            all_velocity_rows.append(np.zeros(8, dtype=np.float32))
            continue        
    
    return np.vstack(all_velocity_rows)

def publish_to_rabbitmq(message, queue_name):
    """Mempublikasikan pesan ke antrian RabbitMQ yang spesifik."""
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
        parameters = pika.ConnectionParameters(
            RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_VHOST, credentials)
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=queue_name, durable=False)
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message,
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()
        print(f"Berhasil mempublikasikan pesan ke antrian '{queue_name}'.")
        return True
    except Exception as e:
        print(f"Error saat publikasi ke antrian '{queue_name}': {e}")
        return False

def parse_filename(filename):
    """Mengekstrak informasi dari nama file."""
    try:
        parts = os.path.basename(filename).replace('.json', '').split('_')
        guid = parts[0]
        timestamp = int(parts[-2])
        index = int(parts[-1])
        return {'guid': guid, 'timestamp': timestamp, 'index': index, 'filename': filename}
    except (IndexError, ValueError):
        return None

def upload_to_ftp(ftp, local_path, remote_filename, remote_folder):
    """Mengunggah satu file ke folder spesifik di server FTP menggunakan koneksi yang ada."""
    try:
        with open(local_path, 'rb') as f:
            # Pastikan direktori tujuan ada
            try:
                ftp.mkd(remote_folder)
            except Exception:
                pass 
            ftp.cwd(remote_folder)
            ftp.storbinary(f'STOR {remote_filename}', f)
        print(f"File '{remote_filename}' berhasil diunggah ke FTP folder '{remote_folder}'.")
        return True
    except Exception as e:
        print(f"Error saat mengunggah '{remote_filename}' ke FTP: {e}")
        return False

# =============================================================================
# EKSEKUSI UTAMA (DAEMON)
# =============================================================================
def main():
    """Loop utama untuk memonitor FTP dan memproses file."""
    for dir_path in [LOCAL_RESULT_DIR, LOCAL_TEMP_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Folder '{dir_path}' telah dibuat.")

    while True:
        batch_processed = False
        print(f"\n[{datetime.now()}] Menghubungkan ke FTP untuk memeriksa folder: '{FTP_SOURCE_FOLDER}'...")
        try:
            with FTP() as ftp:
                ftp.connect(FTP_HOST, FTP_PORT)
                ftp.login(FTP_USER, FTP_PASSWORD)
                ftp.cwd(FTP_SOURCE_FOLDER)

                files_on_ftp = ftp.nlst()
                groups_by_guid = defaultdict(list)
                for filename in files_on_ftp:
                    parsed_info = parse_filename(filename)
                    if parsed_info:
                        groups_by_guid[parsed_info['guid']].append(parsed_info)

                for guid, files in groups_by_guid.items():
                    if len(files) < 8:
                        continue

                    files.sort(key=lambda x: x['timestamp'])

                    for i in range(len(files) - 7):
                        window = files[i : i + 8]
                        
                        time_diff = window[-1]['timestamp'] - window[0]['timestamp']
                        if time_diff <= GROUPING_TIME_WINDOW_MINUTES * 60:
                            indices = {f['index'] for f in window}
                            if indices == set(range(1, 9)):
                                print(f"Batch valid ditemukan untuk GUID {guid} dengan rentang waktu {time_diff} detik.")
                                
                                # Download file batch ke folder temp
                                filenames_to_process = [f['filename'] for f in window]
                                local_paths = []
                                for fname in filenames_to_process:
                                    local_path = os.path.join(LOCAL_TEMP_DIR, fname)
                                    with open(local_path, 'wb') as f_local:
                                        ftp.retrbinary(f'RETR {fname}', f_local.write)
                                    local_paths.append(local_path)
                                print(f"Berhasil mengunduh {len(local_paths)} file ke '{LOCAL_TEMP_DIR}'.")

                                # --- MODIFIKASI PENAMAAN DAN PAYLOAD ---
                                new_uuid = uuid.uuid4()
                                guid_survey = f"SURVEY-{new_uuid}-2025"
                                result_filename = f"{guid_survey}.json"
                                local_result_path = os.path.join(LOCAL_TEMP_DIR, result_filename)
                                
                                velo_matrix = process_batch(local_paths, DIAMETER)
                                velo_list = np.nan_to_num(velo_matrix, posinf=0).tolist()
                                with codecs.open(local_result_path, 'w', encoding='utf-8') as f:
                                    json.dump(velo_list, f, indent=4)
                                
                                # Unggah file hasil ke FTP
                                if upload_to_ftp(ftp, local_result_path, result_filename, FTP_FOLDER_HASIL):
                                    publish_to_rabbitmq(result_filename, RABBITMQ_QUEUE)
                                    
                                    graph_payload = {"GUID_SURVEY": guid_survey, "data": filenames_to_process}
                                    source_files_json = json.dumps(graph_payload)
                                    publish_to_rabbitmq(source_files_json, RABBITMQ_GRAPH_QUEUE)
                                    
                                    # Unggah 8 file sumber ke folder data_row di FTP
                                    for path in local_paths:
                                        upload_to_ftp(ftp, path, os.path.basename(path), FTP_FOLDER_DATA_ROW)

                                    # Hapus file sumber dari FTP folder asal
                                    ftp.cwd(FTP_SOURCE_FOLDER)
                                    for fname in filenames_to_process:
                                        ftp.delete(fname)
                                    print(f"File sumber untuk batch {guid_survey} telah dihapus dari FTP.")

                                # Pindahkan file hasil lokal & bersihkan temp
                                try:
                                    shutil.move(local_result_path, os.path.join(LOCAL_RESULT_DIR, result_filename))
                                    print(f"File hasil '{result_filename}' telah disimpan ke '{LOCAL_RESULT_DIR}'.")
                                    # Hapus file sumber yang diunduh dari temp
                                    for path in local_paths:
                                        os.remove(path)
                                except OSError as e:
                                    print(f"Error saat memindahkan/menghapus file di folder lokal: {e}")
                                
                                batch_processed = True
                                break
                    if batch_processed:
                        break
        except Exception as e:
            print(f"Terjadi error pada loop utama: {e}")

        print(f"Menunggu {PROCESSING_INTERVAL_SECONDS} detik sebelum pengecekan berikutnya...")
        time.sleep(PROCESSING_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
