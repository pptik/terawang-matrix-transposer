#!/usr/bin/env python
# coding: utf-8

import numpy as np
import codecs, json
import time
import os
import pika
import scipy
import uuid 
from ftplib import FTP
from collections import defaultdict
from datetime import datetime
import shutil
from dotenv import load_dotenv
import traceback

# Muat variabel dari file .env
load_dotenv()

# =============================================================================
# KONFIGURASI APLIKASI
# =============================================================================
LOCAL_RESULT_DIR = "hasil_proses_lokal"
LOCAL_TEMP_DIR = "temp_data"
FTP_HOST = os.getenv("FTP_HOST", "ftp-sth.pptik.id")
FTP_PORT = int(os.getenv("FTP_PORT", 2121))
FTP_USER = os.getenv("FTP_USER", "terawang")
FTP_PASSWORD = os.getenv("FTP_PASSWORD", "Terawang@#2025")
FTP_SOURCE_FOLDER = os.getenv("FTP_SOURCE_FOLDER", "/terawang")
FTP_FOLDER_HASIL = os.getenv("FTP_FOLDER_HASIL", "/result")
FTP_FOLDER_DATA_ROW = os.getenv("FTP_FOLDER_DATA_ROW", "/data_row")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rmq230.pptik.id")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME", "terawang")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "Terawang@#2025")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/terawang")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE_RESULT", "result_queue")
RABBITMQ_GRAPH_QUEUE = os.getenv("RABBITMQ_GRAPH_QUEUE", "graph_row")
DIAMETER = 0.3
PROCESSING_INTERVAL_SECONDS = 10
GROUPING_TIME_WINDOW_MINUTES = 15

# =============================================================================
# FUNGSI PEMROSESAN SINYAL (LOGIKA ASLI TIDAK DIUBAH)
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
        # Menggunakan structured array untuk menangani tipe data yang berbeda
        dtype = [(f'value{i}', 'f4') for i in range(1, 9)] + [('timestamp', 'u8')]
        sigarray = np.zeros(min_len, dtype=dtype)
        for i in range(1, 9):
            sigarray[f'value{i}'] = np.array(all_sensor_data[f'value{i}'])[:min_len]
        sigarray['timestamp'] = np.array(timestamp_data)[:min_len]
        return sigarray
    except Exception as e:
        print(f"Error saat menyusun numpy array: {e}")
        return None

def gcc(sig, refsig, fs=1000000, max_tau=None, interp=128, timestamp=None) -> tuple:
    """Menghitung Generalized Cross-Correlation."""
    n = len(sig)
    sig = sig - np.mean(sig, axis=0)
    refsig = refsig - np.mean(refsig, axis=0)
    SIG = np.fft.rfft(sig, axis=0, n=n)
    REFSIG = np.fft.rfft(refsig, axis=0, n=n)
    R = SIG * np.conj(REFSIG)
    WEIGHT = 1 / (np.abs(R) + 1e-10)
    Integ = R * WEIGHT
    cc = np.fft.irfft(Integ, axis=0, n=n)
    lags = scipy.signal.correlation_lags(len(sig), len(refsig), mode= 'same')
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    
    # Menghindari division by zero jika sinyal nol
    max_cc = np.max(cc)
    if max_cc == 0:
        return 0, cc, lags

    smallcc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    smallcc /= max_cc
    shift = np.argmax(smallcc) - max_shift
    cc = scipy.ndimage.shift(cc, len(cc)/2, mode="grid-wrap", order = 5)
    cc /= max_cc
    tau = shift / float(interp * fs)
    if timestamp is not None:
        peaktimestamp = timestamp[np.argmax(cc)]
        timestamp = scipy.ndimage.shift(timestamp, len(timestamp)/2, mode="grid-wrap", order = 5)
        a = timestamp[0]
        b = timestamp[max_shift]
        c = timestamp[-max_shift-1]
        if a > peaktimestamp >=  b:
            tau = int(peaktimestamp - a)
        else:
            tau = int(-peaktimestamp + c)
        tau /= 1000000
    tau /= 10
    return np.abs(tau), cc, lags

def onetap(sigarray: np.ndarray, which: int, diameter: float) -> np.ndarray:
    """Menghitung kecepatan dari satu set data ketukan."""
    try:
        # Mengakses kolom dengan nama dari structured array
        sig1, sig2, sig3, sig4 = sigarray["value1"], sigarray["value2"], sigarray["value3"], sigarray["value4"]
        sig5, sig6, sig7, sig8 = sigarray["value5"], sigarray["value6"], sigarray["value7"], sigarray["value8"]
        timestamp = sigarray["timestamp"]
        
        radius = diameter/2
        ab = radius * 0.76536686473
        ac = radius * 1.41421356237
        ad = radius * 1.84775906502
        ae = float(diameter)

        # Fungsi untuk menghindari pembagian dengan nol
        def safe_div(num, den):
            return num / den if den != 0 else 0

        match which:
            case 1:
                tof12, tof13, tof14, tof15 = gcc(refsig=sig1, sig=sig2, timestamp=timestamp)[0], gcc(refsig=sig1, sig=sig3, timestamp=timestamp)[0], gcc(refsig=sig1, sig=sig4, timestamp=timestamp)[0], gcc(refsig=sig1, sig=sig5, timestamp=timestamp)[0]
                tof16, tof17, tof18 = gcc(refsig=sig1, sig=sig6, timestamp=timestamp)[0], gcc(refsig=sig1, sig=sig7, timestamp=timestamp)[0], gcc(refsig=sig1, sig=sig8, timestamp=timestamp)[0]
                return np.array((0, safe_div(ab, tof12), safe_div(ac, tof13), safe_div(ad, tof14), safe_div(ae, tof15), safe_div(ad, tof16), safe_div(ac, tof17), safe_div(ab, tof18)), dtype=np.float32)
            case 2:
                tof21, tof23, tof24, tof25 = gcc(refsig=sig2, sig=sig1, timestamp=timestamp)[0], gcc(refsig=sig2, sig=sig3, timestamp=timestamp)[0], gcc(refsig=sig2, sig=sig4, timestamp=timestamp)[0], gcc(refsig=sig2, sig=sig5, timestamp=timestamp)[0]
                tof26, tof27, tof28 = gcc(refsig=sig2, sig=sig6, timestamp=timestamp)[0], gcc(refsig=sig2, sig=sig7, timestamp=timestamp)[0], gcc(refsig=sig2, sig=sig8, timestamp=timestamp)[0]
                return np.array((safe_div(ab, tof21), 0, safe_div(ab, tof23), safe_div(ac, tof24), safe_div(ad, tof25), safe_div(ae, tof26), safe_div(ad, tof27), safe_div(ac, tof28)), dtype=np.float32)
            case 3:
                tof31, tof32, tof34, tof35 = gcc(refsig=sig3, sig=sig1, timestamp=timestamp)[0], gcc(refsig=sig3, sig=sig2, timestamp=timestamp)[0], gcc(refsig=sig3, sig=sig4, timestamp=timestamp)[0], gcc(refsig=sig3, sig=sig5, timestamp=timestamp)[0]
                tof36, tof37, tof38 = gcc(refsig=sig3, sig=sig6, timestamp=timestamp)[0], gcc(refsig=sig3, sig=sig7, timestamp=timestamp)[0], gcc(refsig=sig3, sig=sig8, timestamp=timestamp)[0]
                return np.array((safe_div(ac, tof31), safe_div(ab, tof32), 0, safe_div(ab, tof34), safe_div(ac, tof35), safe_div(ad, tof36), safe_div(ae, tof37), safe_div(ad, tof38)), dtype=np.float32)
            case 4:
                tof41, tof42, tof43, tof45 = gcc(refsig=sig4, sig=sig1, timestamp=timestamp)[0], gcc(refsig=sig4, sig=sig2, timestamp=timestamp)[0], gcc(refsig=sig4, sig=sig3, timestamp=timestamp)[0], gcc(refsig=sig4, sig=sig5, timestamp=timestamp)[0]
                tof46, tof47, tof48 = gcc(refsig=sig4, sig=sig6, timestamp=timestamp)[0], gcc(refsig=sig4, sig=sig7, timestamp=timestamp)[0], gcc(refsig=sig4, sig=sig8, timestamp=timestamp)[0]
                return np.array((safe_div(ad, tof41), safe_div(ac, tof42), safe_div(ab, tof43), 0, safe_div(ab, tof45), safe_div(ac, tof46), safe_div(ad, tof47), safe_div(ae, tof48)), dtype=np.float32)
            case 5:
                tof51, tof52, tof53, tof54 = gcc(refsig=sig5, sig=sig1, timestamp=timestamp)[0], gcc(refsig=sig5, sig=sig2, timestamp=timestamp)[0], gcc(refsig=sig5, sig=sig3, timestamp=timestamp)[0], gcc(refsig=sig5, sig=sig4, timestamp=timestamp)[0]
                tof56, tof57, tof58 = gcc(refsig=sig5, sig=sig6, timestamp=timestamp)[0], gcc(refsig=sig5, sig=sig7, timestamp=timestamp)[0], gcc(refsig=sig5, sig=sig8, timestamp=timestamp)[0]
                return np.array((safe_div(ae, tof51), safe_div(ad, tof52), safe_div(ac, tof53), safe_div(ab, tof54), 0, safe_div(ab, tof56), safe_div(ac, tof57), safe_div(ad, tof58)), dtype=np.float32)
            case 6:
                tof61, tof62, tof63, tof64 = gcc(refsig=sig6, sig=sig1, timestamp=timestamp)[0], gcc(refsig=sig6, sig=sig2, timestamp=timestamp)[0], gcc(refsig=sig6, sig=sig3, timestamp=timestamp)[0], gcc(refsig=sig6, sig=sig4, timestamp=timestamp)[0]
                tof65, tof67, tof68 = gcc(refsig=sig6, sig=sig5, timestamp=timestamp)[0], gcc(refsig=sig6, sig=sig7, timestamp=timestamp)[0], gcc(refsig=sig6, sig=sig8, timestamp=timestamp)[0]
                return np.array((safe_div(ad, tof61), safe_div(ae, tof62), safe_div(ad, tof63), safe_div(ac, tof64), safe_div(ab, tof65), 0, safe_div(ab, tof67), safe_div(ac, tof68)), dtype=np.float32)
            case 7:
                tof71, tof72, tof73, tof74 = gcc(refsig=sig7, sig=sig1, timestamp=timestamp)[0], gcc(refsig=sig7, sig=sig2, timestamp=timestamp)[0], gcc(refsig=sig7, sig=sig3, timestamp=timestamp)[0], gcc(refsig=sig7, sig=sig4, timestamp=timestamp)[0]
                tof75, tof76, tof78 = gcc(refsig=sig7, sig=sig5, timestamp=timestamp)[0], gcc(refsig=sig7, sig=sig6, timestamp=timestamp)[0], gcc(refsig=sig7, sig=sig8, timestamp=timestamp)[0]
                return np.array((safe_div(ac, tof71), safe_div(ad, tof72), safe_div(ae, tof73), safe_div(ad, tof74), safe_div(ac, tof75), safe_div(ab, tof76), 0, safe_div(ab, tof78)), dtype=np.float32)
            case 8:
                tof81, tof82, tof83, tof84 = gcc(refsig=sig8, sig=sig1, timestamp=timestamp)[0], gcc(refsig=sig8, sig=sig2, timestamp=timestamp)[0], gcc(refsig=sig8, sig=sig3, timestamp=timestamp)[0], gcc(refsig=sig8, sig=sig4, timestamp=timestamp)[0]
                tof85, tof86, tof87 = gcc(refsig=sig8, sig=sig5, timestamp=timestamp)[0], gcc(refsig=sig8, sig=sig6, timestamp=timestamp)[0], gcc(refsig=sig8, sig=sig7, timestamp=timestamp)[0]
                return np.array((safe_div(ab, tof81), safe_div(ac, tof82), safe_div(ad, tof83), safe_div(ae, tof84), safe_div(ad, tof85), safe_div(ac, tof86), safe_div(ab, tof87), 0), dtype=np.float32)
            case _:
                raise ValueError("Invalid number. Expected between 1 and 8")
    except Exception as e:
        print(f"Error saat menghitung ToF: {e}")
        return np.zeros(8, dtype=np.float32)

# =============================================================================
# FUNGSI UTILITAS
# =============================================================================
def process_batch(file_paths, diameter):
    """Memproses satu batch (8 file) dan mengembalikan matriks kecepatan."""
    all_velocity_rows = []
    
    def get_index_from_path(path):
        try:
            return int(os.path.basename(path).split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            return 0
            
    sorted_paths = sorted(file_paths, key=get_index_from_path)
    
    for i, filepath in enumerate(sorted_paths, start=1):
        sigarray = load_sigarray_from_json(filepath)
        if sigarray is not None:
            velocity_row = onetap(sigarray, which=i, diameter=diameter)
            all_velocity_rows.append(velocity_row)
        else:
            print(f"Gagal memproses {os.path.basename(filepath)}, baris akan diisi nol.")
            all_velocity_rows.append(np.zeros(8, dtype=np.float32))
            
    return np.vstack(all_velocity_rows) if all_velocity_rows else np.array([])


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
            properties=pika.BasicProperties(delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
        )
        connection.close()
        print(f"Pesan berhasil dipublikasikan ke antrian '{queue_name}'.")
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
    """Mengunggah satu file ke folder spesifik di server FTP."""
    try:
        ftp.cwd("/")
        try:
            ftp.mkd(remote_folder)
        except Exception:
            pass 
        ftp.cwd(remote_folder)
        with open(local_path, 'rb') as f:
            ftp.storbinary(f'STOR {remote_filename}', f)
        print(f"File '{remote_filename}' berhasil diunggah ke '{remote_folder}'.")
        return True
    except Exception as e:
        print(f"Error saat mengunggah '{remote_filename}' ke FTP: {e}")
        return False

# =============================================================================
# EKSEKUSI UTAMA (DAEMON) - DENGAN PERBAIKAN
# =============================================================================
def main():
    """Loop utama untuk memonitor FTP dan memproses file."""
    for dir_path in [LOCAL_RESULT_DIR, LOCAL_TEMP_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Folder '{dir_path}' telah dibuat.")

    while True:
        batch_processed = False
        print(f"\n[{datetime.now()}] Menghubungkan ke FTP: '{FTP_SOURCE_FOLDER}'...")
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
                                print(f"✅ Batch valid ditemukan untuk GUID {guid} (rentang waktu {time_diff}s).")
                                
                                # ---- PERBAIKAN UTAMA ----
                                # 1. Siapkan daftar file asli (untuk FTP) dan nama aman (untuk lokal)
                                original_filenames = [f['filename'] for f in window]
                                safe_local_paths = []
                                downloaded_files_map = {} # Mapping nama asli ke path lokal aman

                                # 2. Unduh file dengan nama yang aman
                                print("  Mengunduh file batch...")
                                for ftp_fname in original_filenames:
                                    # Ganti karakter ilegal ':' dengan '-' untuk path lokal
                                    safe_fname = ftp_fname.replace(':', '-')
                                    local_path = os.path.join(LOCAL_TEMP_DIR, safe_fname)
                                    
                                    with open(local_path, 'wb') as f_local:
                                        ftp.retrbinary(f'RETR {ftp_fname}', f_local.write)
                                    
                                    safe_local_paths.append(local_path)
                                    downloaded_files_map[ftp_fname] = local_path
                                print(f"  Berhasil mengunduh {len(safe_local_paths)} file.")

                                # 3. Proses batch menggunakan file lokal yang sudah aman
                                new_uuid = uuid.uuid4()
                                guid_survey = f"SURVEY-{new_uuid}-2025"
                                result_filename = f"{guid_survey}.json"
                                local_result_path = os.path.join(LOCAL_TEMP_DIR, result_filename)
                                
                                velo_matrix = process_batch(safe_local_paths, DIAMETER)
                                velo_list = np.nan_to_num(velo_matrix, posinf=0, neginf=0).tolist()
                                with codecs.open(local_result_path, 'w', encoding='utf-8') as f:
                                    json.dump(velo_list, f, indent=4)
                                
                                # 4. Unggah, kirim pesan, dan hapus menggunakan nama yang sesuai
                                if upload_to_ftp(ftp, local_result_path, result_filename, FTP_FOLDER_HASIL):
                                    publish_to_rabbitmq(result_filename, RABBITMQ_QUEUE)
                                    
                                    graph_payload = {"GUID_SURVEY": guid_survey, "data": original_filenames}
                                    publish_to_rabbitmq(json.dumps(graph_payload), RABBITMQ_GRAPH_QUEUE)
                                    
                                    # Unggah 8 file sumber (gunakan path lokal aman, nama remote asli)
                                    for ftp_fname, local_path in downloaded_files_map.items():
                                        upload_to_ftp(ftp, local_path, ftp_fname, FTP_FOLDER_DATA_ROW)

                                    # Hapus file sumber dari FTP (gunakan nama asli)
                                    ftp.cwd(FTP_SOURCE_FOLDER)
                                    for ftp_fname in original_filenames:
                                        ftp.delete(ftp_fname)
                                    print(f"  File sumber untuk batch {guid_survey} telah dipindahkan.")

                                # 5. Pindahkan file hasil dan bersihkan folder temp
                                shutil.move(local_result_path, os.path.join(LOCAL_RESULT_DIR, result_filename))
                                for path in safe_local_paths:
                                    os.remove(path)
                                print(f"  Hasil '{result_filename}' disimpan & folder temp dibersihkan.")
                                
                                batch_processed = True
                                break # Keluar dari loop 'for i'
                    if batch_processed:
                        break # Keluar dari loop 'for guid'

        except Exception as e:
            print(f"❌ Terjadi error pada loop utama: {e}")
            traceback.print_exc()

        print(f"Menunggu {PROCESSING_INTERVAL_SECONDS} detik...")
        time.sleep(PROCESSING_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()