#!/usr/bin/env python
# coding: utf-8

import numpy as np
import codecs, json
import time
import os
import pika
import scipy
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

def gcc(sig, refsig, fs=1000000, max_tau=None, interp=128, timestamp=None) -> tuple:
    """Menghitung Generalized Cross-Correlation."""
    
    # Generalized Cross Correlation Phase Transform
    n = len(sig)
    
    # Remove DC component
    sig = sig - np.mean(sig, axis=0)
    refsig = refsig - np.mean(refsig, axis=0)
    
    # RFFT because it's faster, it doesn't compute the negative side
    SIG = np.fft.rfft(sig, axis=0, n=n)
    REFSIG = np.fft.rfft(refsig, axis=0, n=n)
    R = SIG * np.conj(REFSIG)
    
    WEIGHT = 1 / (np.abs(R) + 1e-10) # No need to use anything else other than PHAT
    
    Integ = R * WEIGHT
    cc = np.fft.irfft(Integ, axis=0, n=n)
    lags = scipy.signal.correlation_lags(len(sig), len(refsig), mode= 'same')

    max_shift = int(interp * n / 2)
    
    if max_tau:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    smallcc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    smallcc /= np.max(cc)
    
    # find max cross correlation index
    shift = np.argmax(smallcc) - max_shift
    
    # Sometimes, there is a 180-degree phase difference between the two microphones.
    # shift = np.argmax(np.abs(cc)) - max_shift
    
    cc = scipy.ndimage.shift(cc, len(cc)/2, mode="grid-wrap", order = 5)
    cc /= np.max(cc)
    
    tau = shift / float(interp * fs)
    
    if timestamp is not None:
        
        peaktimestamp = timestamp[np.argmax(cc)]
        
        timestamp = scipy.ndimage.shift(timestamp, len(timestamp)/2, mode="grid-wrap", order = 5)
        
        a = timestamp[0] # first possible timestamp on the dataframe
        b = timestamp[max_shift] # timestamp that corresponds fo the end of smalltimestamp 
        c = timestamp[-max_shift-1] # timestamp that corresponds to the start of the smalltimestamp
        d = timestamp[-1] # last possible timestamp on the dataframe
        # smalltimestamp = np.concatenate((timestamp[-max_shift:], timestamp[:max_shift+1]))
        # peaktimestamp = smalltimestamp[np.argmax(smallcc)]
        
        if a > peaktimestamp >=  b:
            tau = int(peaktimestamp - a) # in micros
        else:
            tau = int(-peaktimestamp + c) # in micros, negative
        tau /= 1000000 # convert to seconds

    tau /= 10
    
    return np.abs(tau), cc, lags

def onetap(sigarray: np.ndarray, which: int, diameter: float) -> np.ndarray:
    """Menghitung kecepatan dari satu set data ketukan."""
    
    # function to tap once. produces 7 ToF/tau from 7 CC, out of 8 sensors
    
    # diameters in meters
    
    sig1 = sigarray["value1"]
    sig2 = sigarray["value2"]
    sig3 = sigarray["value3"]
    sig4 = sigarray["value4"]
    sig5 = sigarray["value5"]
    sig6 = sigarray["value6"]
    sig7 = sigarray["value7"]
    sig8 = sigarray["value8"]
    timestamp = sigarray["timestamp"]
    
    radius = diameter/2
    ab = radius * 0.76536686473 # sqrt(sqrt(2)-2)
    ac = radius * 1.41421356237 # sqrt(2)
    ad = radius * 1.84775906502 # sqrt(sqrt(2)+2)
    ae = float(diameter)
    
    # ab = 12,23,34,45,56,67,78,81
    # ac = 13,24,35,46,57,68,71,82
    # ad = 14,25,36,47,58,61,72,83
    # ae = 15,26,37,48,51,62,73,84
    
    try:
        match which:
            case 1:
                tof12 = gcc(refsig=sig1, sig=sig2, timestamp=timestamp)[0]
                tof13 = gcc(refsig=sig1, sig=sig3, timestamp=timestamp)[0]
                tof14 = gcc(refsig=sig1, sig=sig4, timestamp=timestamp)[0]
                tof15 = gcc(refsig=sig1, sig=sig5, timestamp=timestamp)[0]
                tof16 = gcc(refsig=sig1, sig=sig6, timestamp=timestamp)[0]
                tof17 = gcc(refsig=sig1, sig=sig7, timestamp=timestamp)[0]
                tof18 = gcc(refsig=sig1, sig=sig8, timestamp=timestamp)[0]
                velo12 = ab / tof12
                velo13 = ac / tof13
                velo14 = ad / tof14
                velo15 = ae / tof15
                velo16 = ad / tof16
                velo17 = ac / tof17
                velo18 = ab / tof18
            
                return np.array((0, velo12, velo13, velo14, velo15, velo16, velo17, velo18), dtype=np.float32)
            case 2:
                tof21 = gcc(refsig=sig2, sig=sig1, timestamp=timestamp)[0]
                tof23 = gcc(refsig=sig2, sig=sig3, timestamp=timestamp)[0]
                tof24 = gcc(refsig=sig2, sig=sig4, timestamp=timestamp)[0]
                tof25 = gcc(refsig=sig2, sig=sig5, timestamp=timestamp)[0]
                tof26 = gcc(refsig=sig2, sig=sig6, timestamp=timestamp)[0]
                tof27 = gcc(refsig=sig2, sig=sig7, timestamp=timestamp)[0]
                tof28 = gcc(refsig=sig2, sig=sig8, timestamp=timestamp)[0]
                velo21 = ab / tof21
                velo23 = ab / tof23
                velo24 = ac / tof24
                velo25 = ad / tof25
                velo26 = ae / tof26
                velo27 = ad / tof27
                velo28 = ac / tof28
            
                return np.array((velo21, 0, velo23, velo24, velo25, velo26, velo27, velo28), dtype=np.float32)
            case 3:
                tof31 = gcc(refsig=sig3, sig=sig1, timestamp=timestamp)[0]
                tof32 = gcc(refsig=sig3, sig=sig2, timestamp=timestamp)[0]
                tof34 = gcc(refsig=sig3, sig=sig4, timestamp=timestamp)[0]
                tof35 = gcc(refsig=sig3, sig=sig5, timestamp=timestamp)[0]
                tof36 = gcc(refsig=sig3, sig=sig6, timestamp=timestamp)[0]
                tof37 = gcc(refsig=sig3, sig=sig7, timestamp=timestamp)[0]
                tof38 = gcc(refsig=sig3, sig=sig8, timestamp=timestamp)[0]
                velo31 = ac / tof31
                velo32 = ab / tof32
                velo34 = ab / tof34
                velo35 = ac / tof35
                velo36 = ad / tof36
                velo37 = ae / tof37
                velo38 = ad / tof38
            
                return np.array((velo31, velo32, 0, velo34, velo35, velo36, velo37, velo38), dtype=np.float32) 
            case 4:
                tof41 = gcc(refsig=sig4, sig=sig1, timestamp=timestamp)[0]
                tof42 = gcc(refsig=sig4, sig=sig2, timestamp=timestamp)[0]
                tof43 = gcc(refsig=sig4, sig=sig3, timestamp=timestamp)[0]
                tof45 = gcc(refsig=sig4, sig=sig5, timestamp=timestamp)[0]
                tof46 = gcc(refsig=sig4, sig=sig6, timestamp=timestamp)[0]
                tof47 = gcc(refsig=sig4, sig=sig7, timestamp=timestamp)[0]
                tof48 = gcc(refsig=sig4, sig=sig8, timestamp=timestamp)[0]
                velo41 = ad / tof41
                velo42 = ac / tof42
                velo43 = ab / tof43
                velo45 = ab / tof45
                velo46 = ac / tof46
                velo47 = ad / tof47
                velo48 = ae / tof48
            
                return np.array((velo41, velo42, velo43, 0, velo45, velo46, velo47, velo48), dtype=np.float32)
            case 5:
                tof51 = gcc(refsig=sig5, sig=sig1, timestamp=timestamp)[0]
                tof52 = gcc(refsig=sig5, sig=sig2, timestamp=timestamp)[0]
                tof53 = gcc(refsig=sig5, sig=sig3, timestamp=timestamp)[0]
                tof54 = gcc(refsig=sig5, sig=sig4, timestamp=timestamp)[0]
                tof56 = gcc(refsig=sig5, sig=sig6, timestamp=timestamp)[0]
                tof57 = gcc(refsig=sig5, sig=sig7, timestamp=timestamp)[0]
                tof58 = gcc(refsig=sig5, sig=sig8, timestamp=timestamp)[0]
                velo51 = ae / tof51
                velo52 = ad / tof52
                velo53 = ac / tof53
                velo54 = ab / tof54
                velo56 = ab / tof56
                velo57 = ac / tof57
                velo58 = ad / tof58
            
                return np.array((velo51, velo52, velo53, velo54, 0, velo56, velo57, velo58), dtype=np.float32)
            case 6:
                tof61 = gcc(refsig=sig6, sig=sig1, timestamp=timestamp)[0]
                tof62 = gcc(refsig=sig6, sig=sig2, timestamp=timestamp)[0]
                tof63 = gcc(refsig=sig6, sig=sig3, timestamp=timestamp)[0]
                tof64 = gcc(refsig=sig6, sig=sig4, timestamp=timestamp)[0]
                tof65 = gcc(refsig=sig6, sig=sig5, timestamp=timestamp)[0]
                tof67 = gcc(refsig=sig6, sig=sig7, timestamp=timestamp)[0]
                tof68 = gcc(refsig=sig6, sig=sig8, timestamp=timestamp)[0]
                velo61 = ad / tof61
                velo62 = ae / tof62
                velo63 = ad / tof63
                velo64 = ac / tof64
                velo65 = ab / tof65
                velo67 = ab / tof67
                velo68 = ac / tof68
            
                return np.array((velo61, velo62, velo63, velo64, velo65, 0, velo67, velo68), dtype=np.float32)
            case 7:
                tof71 = gcc(refsig=sig7, sig=sig1, timestamp=timestamp)[0]
                tof72 = gcc(refsig=sig7, sig=sig2, timestamp=timestamp)[0]
                tof73 = gcc(refsig=sig7, sig=sig3, timestamp=timestamp)[0]
                tof74 = gcc(refsig=sig7, sig=sig4, timestamp=timestamp)[0]
                tof75 = gcc(refsig=sig7, sig=sig5, timestamp=timestamp)[0]
                tof76 = gcc(refsig=sig7, sig=sig6, timestamp=timestamp)[0]
                tof78 = gcc(refsig=sig7, sig=sig8, timestamp=timestamp)[0]
                velo71 = ac / tof71
                velo72 = ad / tof72
                velo73 = ae / tof73
                velo74 = ad / tof74
                velo75 = ac / tof75
                velo76 = ab / tof76
                velo78 = ab / tof78
            
                return np.array((velo71, velo72, velo73, velo74, velo75, velo76, 0, velo78), dtype=np.float32)
            case 8:
                tof81 = gcc(refsig=sig8, sig=sig1, timestamp=timestamp)[0]
                tof82 = gcc(refsig=sig8, sig=sig2, timestamp=timestamp)[0]
                tof83 = gcc(refsig=sig8, sig=sig3, timestamp=timestamp)[0]
                tof84 = gcc(refsig=sig8, sig=sig4, timestamp=timestamp)[0]
                tof85 = gcc(refsig=sig8, sig=sig5, timestamp=timestamp)[0]
                tof86 = gcc(refsig=sig8, sig=sig6, timestamp=timestamp)[0]
                tof87 = gcc(refsig=sig8, sig=sig7, timestamp=timestamp)[0]
                velo81 = ab / tof81
                velo82 = ac / tof82
                velo83 = ad / tof83
                velo84 = ae / tof84
                velo85 = ad / tof85
                velo86 = ac / tof86
                velo87 = ab / tof87
            
                return np.array((velo81, velo82, velo83, velo84, velo85, velo86, velo87, 0), dtype=np.float32)
            case _:
                raise ValueError("Invalid number. Expected between 1 and 8")
    except ValueError as ve:
        print(f"Error saat menghitung ToF: {ve}")
        return np.zeros(8, dtype=np.float32)

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
                                # You don't need a nan_to_num if you make your code correctly
                                jsonpayload = {"species":"Mahoni","keliling":155.8,"jumlah_sensor":8,"ketinggian":1,"data_pengukuran": velo_matrix}
                                with codecs.open(local_result_path, 'w', encoding='utf-8') as f:
                                    json.dumps(jsonpayload)
                                
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
