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
from gccestimating import GCC, corrlags

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


def gccest(siga, sigb, samplerate=1, cctype="phat"):
    n = int((len(siga)+ len(sigb)) / 2)
    
    siga -= np.mean(siga, axis=0)
    sigb -= np.mean(sigb, axis=0)
    
    lags = corrlags(2*n-1, samplerate=samplerate)
    
    gcc = GCC(sig1=siga,sig2=sigb)
    
    match cctype.lower():
        case "cc":
            cc = gcc.cc()
        case "phat":
            cc = gcc.phat()
        case "scot":
            cc = gcc.scot()
        case"roth":
            cc = gcc.roth()
        case"ht":
            cc = gcc.ht()
        case _:
            cc = gcc.cc()
    
    cc /= np.max(np.abs(cc))    # normalize

    return cc, lags

# y = 27808x + timestamp[0]

def timestampextrapolate(x,origin):
    return 27808*x + origin

def tauest(cc, lags, samplerate = 1, timestamp = None):
    shift = np.argmax(np.abs(cc)) # the index of the maximum value in cc
    # no need to roll to center, it is already centered
    tau = lags[shift] / float(samplerate)  # in seconds
    if timestamp is not None:
        if len(timestamp) < 999:
            # append to timestamp
            for x in range(1,251,1): # x from 1 to 250
                timestamp = np.append(timestamp, timestampextrapolate(x=x,origin=timestamp[0]))
            for x in range(-1,-250,-1): # x from -1 to -249
                timestamp = np.insert(timestamp, 0, timestampextrapolate(x=x,origin=timestamp[0]))     
        peaktimestamp = timestamp[shift]
        origintimestamp = timestamp[np.argmin(np.abs(lags))]  # the timestamp corresponding to the zero lag
        tau = peaktimestamp - origintimestamp # in nanoseconds
        # timestamp needs to be extrapolated to 2x the length
        tau /= 1000000000 # convert to seconds
    return np.abs(tau), shift

def safe_divide(num, denom, default_value=1e8):
    if not isinstance(num, (int, float)) or not isinstance(denom, (int, float)):
        raise ValueError("Both num and denom must be numbers!")
    
    if denom == 0:
        return default_value
    
    return num / denom

def taufromsig(siga, sigb, samplerate= 1, timestamp= None):
    cc, lags = gccest(siga= siga, sigb= sigb, samplerate= samplerate)
    tau, shift = tauest(cc= cc, lags= lags, samplerate= samplerate, timestamp= timestamp)
    return tau, shift

def onetap(sigarray: np.ndarray, which: int, diameter = 0.3, samplerate = 35961):
    
    # Samples per second
    # Why? it takes an average of 27808 nanoseconds to capture one sample
    # so 1 second / 27808 nanoseconds = 35960.8746 samples per second
    # Round up to 35961 samples per second
    
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
    
    match which:
        case 1:
            tof12 = taufromsig(siga= sig1, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo12 = safe_divide(ab, tof12)
            
            tof13 = taufromsig(siga= sig1, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo13 = safe_divide(ac, tof13)
            
            tof14 = taufromsig(siga= sig1, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo14 = safe_divide(ad, tof14)
            
            tof15 = taufromsig(siga= sig1, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo15 = safe_divide(ae, tof15)
            
            tof16 = taufromsig(siga= sig1, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo16 = safe_divide(ad, tof16)
            
            tof17 = taufromsig(siga= sig1, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo17 = safe_divide(ac, tof17)
            
            tof18 = taufromsig(siga= sig1, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo18 = safe_divide(ab, tof18)
            
            return np.array((0, velo12, velo13, velo14, velo15, velo16, velo17, velo18), dtype=np.float32)
        case 2:
            
            tof21 = taufromsig(siga= sig2, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo21 = safe_divide(ab, tof21)
            
            tof23 = taufromsig(siga= sig2, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo23 = safe_divide(ab, tof23)
            
            tof24 = taufromsig(siga= sig2, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo24 = safe_divide(ac, tof24)
            
            tof25 = taufromsig(siga= sig2, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo25 = safe_divide(ad, tof25)
            
            tof26 = taufromsig(siga= sig2, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo26 = safe_divide(ae, tof26)
            
            tof27 = taufromsig(siga= sig2, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo27 = safe_divide(ad, tof27)
            
            tof28 = taufromsig(siga= sig2, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo28 = safe_divide(ac, tof28)
            
            return np.array((velo21, 0, velo23, velo24, velo25, velo26, velo27, velo28), dtype=np.float32)
        case 3:
            
            tof31 = taufromsig(siga= sig3, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo31 = safe_divide(ac, tof31)
            
            tof32 = taufromsig(siga= sig3, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo32 = safe_divide(ab, tof32)
            
            tof34 = taufromsig(siga= sig3, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo34 = safe_divide(ab, tof34)
            
            tof35 = taufromsig(siga= sig3, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo35 = safe_divide(ac, tof35)
            
            tof36 = taufromsig(siga= sig3, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo36 = safe_divide(ad, tof36)
            
            tof37 = taufromsig(siga= sig3, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo37 = safe_divide(ae, tof37)
            
            tof38 = taufromsig(siga= sig3, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo38 = safe_divide(ad, tof38)
            
            return np.array((velo31, velo32, 0, velo34, velo35, velo36, velo37, velo38), dtype=np.float32) 
        case 4:
            
            tof41 = taufromsig(siga= sig4, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo41 = safe_divide(ad, tof41)
            
            tof42 = taufromsig(siga= sig4, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo42 = safe_divide(ac, tof42)
            
            tof43 = taufromsig(siga= sig4, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo43 = safe_divide(ab, tof43)
            
            tof45 = taufromsig(siga= sig4, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo45 = safe_divide(ab, tof45)
            
            tof46 = taufromsig(siga= sig4, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo46 = safe_divide(ac, tof46)
            
            tof47 = taufromsig(siga= sig4, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo47 = safe_divide(ad, tof47)
            
            tof48 = taufromsig(siga= sig4, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo48 = safe_divide(ae, tof48)
            
            return np.array((velo41, velo42, velo43, 0, velo45, velo46, velo47, velo48), dtype=np.float32)
        case 5:
            
            tof51 = taufromsig(siga= sig5, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo51 = safe_divide(ae, tof51)
            
            tof52 = taufromsig(siga= sig5, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo52 = safe_divide(ad, tof52)
            
            tof53 = taufromsig(siga= sig5, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo53 = safe_divide(ac, tof53)
            
            tof54 = taufromsig(siga= sig5, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo54 = safe_divide(ab, tof54)
            
            tof56 = taufromsig(siga= sig5, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo56 = safe_divide(ab, tof56)
            
            tof57 = taufromsig(siga= sig5, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo57 = safe_divide(ac, tof57)
            
            tof58 = taufromsig(siga= sig5, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo58 = safe_divide(ad, tof58)
            
            return np.array((velo51, velo52, velo53, velo54, 0, velo56, velo57, velo58), dtype=np.float32)
        case 6:
            
            tof61 = taufromsig(siga= sig6, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo61 = safe_divide(ad, tof61)
            
            tof62 = taufromsig(siga= sig6, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo62 = safe_divide(ae, tof62)
            
            tof63 = taufromsig(siga= sig6, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo63 = safe_divide(ad, tof63)
            
            tof64 = taufromsig(siga= sig6, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo64 = safe_divide(ac, tof64)
            
            tof65 = taufromsig(siga= sig6, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo65 = safe_divide(ab, tof65)
            
            tof67 = taufromsig(siga= sig6, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo67 = safe_divide(ab, tof67)
            
            tof68 = taufromsig(siga= sig6, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo68 = safe_divide(ac, tof68)
            
            return np.array((velo61, velo62, velo63, velo64, velo65, 0, velo67, velo68), dtype=np.float32)
        case 7:
            
            tof71 = taufromsig(siga= sig7, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo71 = safe_divide(ac, tof71)
            
            tof72 = taufromsig(siga= sig7, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo72 = safe_divide(ad, tof72)
            
            tof73 = taufromsig(siga= sig7, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo73 = safe_divide(ae, tof73)
            
            tof74 = taufromsig(siga= sig7, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo74 = safe_divide(ad, tof74)
            
            tof75 = taufromsig(siga= sig7, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo75 = safe_divide(ac, tof75)
            
            tof76 = taufromsig(siga= sig7, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo76 = safe_divide(ab, tof76)
            
            tof78 = taufromsig(siga= sig7, sigb= sig8, samplerate= samplerate, timestamp= timestamp)[0]
            velo78 = safe_divide(ab, tof78)
            
            return np.array((velo71, velo72, velo73, velo74, velo75, velo76, 0, velo78), dtype=np.float32)
        case 8:
            tof81 = taufromsig(siga= sig8, sigb= sig1, samplerate= samplerate, timestamp= timestamp)[0]
            velo81 = safe_divide(ab, tof81)
            
            tof82 = taufromsig(siga= sig8, sigb= sig2, samplerate= samplerate, timestamp= timestamp)[0]
            velo82 = safe_divide(ac, tof82)
            
            tof83 = taufromsig(siga= sig8, sigb= sig3, samplerate= samplerate, timestamp= timestamp)[0]
            velo83 = safe_divide(ad, tof83)
            
            tof84 = taufromsig(siga= sig8, sigb= sig4, samplerate= samplerate, timestamp= timestamp)[0]
            velo84 = safe_divide(ae, tof84)
            
            tof85 = taufromsig(siga= sig8, sigb= sig5, samplerate= samplerate, timestamp= timestamp)[0]
            velo85 = safe_divide(ad, tof85)
            
            tof86 = taufromsig(siga= sig8, sigb= sig6, samplerate= samplerate, timestamp= timestamp)[0]
            velo86 = safe_divide(ac, tof86)
            
            tof87 = taufromsig(siga= sig8, sigb= sig7, samplerate= samplerate, timestamp= timestamp)[0]
            velo87 = safe_divide(ab, tof87)
            
            return np.array((velo81, velo82, velo83, velo84, velo85, velo86, velo87, 0), dtype=np.float32)
        case _:
            raise ValueError

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