import numpy as np
import json
import codecs
from gccestimating import GCC, corrlags
import matplotlib.pyplot as plt
import pika
import time
import os
import sys
import re
import uuid
from ftplib import FTP
from dotenv import load_dotenv
from datetime import datetime

# ==============================================================================
# KONFIGURASI DAN VARIABEL LINGKUNGAN
# ==============================================================================

load_dotenv()

# --- KONFIGURASI RABBITMQ ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT"))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST")
RABBITMQ_CONSUME_QUEUE = 'terawang-folder' # Queue untuk menerima folder
RABBITMQ_PUBLISH_QUEUE = 'terawangDataTof' # Queue baru untuk mengirim hasil

# --- KONFIGURASI FTP (dari file .env) ---
FTP_HOST = os.getenv("FTP_HOST")
FTP_PORT = int(os.getenv("FTP_PORT", 21))
FTP_USER = os.getenv("FTP_USER")
FTP_PASSWORD = os.getenv("FTP_PASSWORD")
FTP_RESULT_FOLDER = os.getenv("FTP_SOURCE_FOLDER", "/terawang") # Default ke /terawang jika tidak diset

# ==============================================================================
# FUNGSI-FUNGSI BANTUAN (FTP & RMQ PUBLISH)
# ==============================================================================

def upload_to_ftp(local_filepath, remote_filename):
    """Mengunggah file dari path lokal ke server FTP."""
    if not all([FTP_HOST, FTP_USER, FTP_PASSWORD, FTP_RESULT_FOLDER]):
        print("[!] Peringatan: Konfigurasi FTP tidak lengkap. Proses unggah dilewati.")
        return False
        
    try:
        with FTP() as ftp:
            print(f"    -> Menghubungkan ke FTP di {FTP_HOST}...")
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASSWORD)
            
            # Coba ganti direktori, buat jika belum ada
            try:
                ftp.cwd(FTP_RESULT_FOLDER)
            except Exception:
                print(f"    -> Direktori '{FTP_RESULT_FOLDER}' tidak ditemukan, mencoba membuatnya...")
                ftp.mkd(FTP_RESULT_FOLDER)
                ftp.cwd(FTP_RESULT_FOLDER)

            with open(local_filepath, 'rb') as file:
                print(f"    -> Mengunggah {os.path.basename(local_filepath)} sebagai {remote_filename}...")
                ftp.storbinary(f'STOR {remote_filename}', file)
            
            print(f"    -> Unggah FTP berhasil: {remote_filename}")
            return True
    except Exception as e:
        print(f"[!] Error saat mengunggah ke FTP: {e}")
        return False

def publish_result_to_rmq(channel, queue_name, json_data):
    """Mempublikasikan data JSON ke antrian RabbitMQ yang ditentukan."""
    try:
        # Pastikan antrian tujuan ada (durable=True agar tidak hilang saat restart)
        channel.queue_declare(queue=queue_name, durable=True)
        
        # Konversi dictionary Python ke string JSON
        message_body = json.dumps(json_data, indent=4)
        
        # Publikasikan pesan
        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=message_body,
            properties=pika.BasicProperties(
                delivery_mode=2,  # Membuat pesan persistent
            )
        )
        print(f"[*] Berhasil mempublikasikan hasil ke antrian RMQ: '{queue_name}'")
        return True
    except Exception as e:
        print(f"[!] Gagal mempublikasikan hasil ke RMQ: {e}")
        return False
        
# ==============================================================================
# SEMUA FUNGSI PERHITUNGAN INTI (TIDAK DIUBAH)
# ==============================================================================
def loadcsv(filename, delim=","):
    data = np.loadtxt(filename, delimiter=delim, dtype= np.float64)
    return data

def loadjson(filename):
    print(f"    -> Membaca file: {os.path.basename(filename)}")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"    -> ERROR: File tidak ditemukan di {filename}")
        return None
    except json.JSONDecodeError:
        print(f"    -> ERROR: Format JSON tidak valid di {filename}")
        return None

def gccest(siga, sigb, samplerate=1, cctype="scot"):
    n = int((len(siga)+ len(sigb)) / 2)
    siga -= np.mean(siga, axis=0)
    sigb -= np.mean(sigb, axis=0)
    lags = corrlags(2*n-1, samplerate=samplerate)
    gcc = GCC(sig1=siga,sig2=sigb)
    
    match cctype.lower():
        case "cc": cc = gcc.cc()
        case "phat": cc = gcc.phat()
        case "scot": cc = gcc.scot()
        case "roth": cc = gcc.roth()
        case "ht": cc = gcc.ht()
        case _: cc = gcc.cc()
    
    cc /= np.max(np.abs(cc))
    return cc, lags

def findtimestampavg(arr):
    return np.mean(np.diff(arr))

def timestampextrapfromavg(x,origin,arr):
    return findtimestampavg(arr)*x + origin

def tauest(cc, lags, samplerate = 1, timestamp = None):
    shift = np.argmax(np.abs(cc))
    tau = lags[shift] / float(samplerate)
    if timestamp is not None:
        if len(timestamp) < 999:
            for x in range(1,251,1):
                timestamp = np.append(timestamp, timestampextrapfromavg(x=x,origin=timestamp[0], arr=timestamp))
            for x in range(-1,-250,-1):
                timestamp = np.insert(timestamp, 0, timestampextrapfromavg(x=x,origin=timestamp[0], arr=timestamp))
        peaktimestamp = timestamp[shift]
        origintimestamp = timestamp[np.argmin(np.abs(lags))]
        tau = peaktimestamp - origintimestamp
        tau /= 1000000000
    return np.abs(tau), shift

def safe_divide(num, denom, default_value=0):
    if not isinstance(num, (int, float)) or not isinstance(denom, (int, float)):
        raise ValueError("Both num and denom must be numbers!")
    else:
        if denom == 0:
            return default_value
        else:
            return num / denom

def taufromsig(siga, sigb, samplerate= 1, timestamp= None):
    cc, lags = gccest(siga= siga, sigb= sigb, samplerate= samplerate)
    tau, shift = tauest(cc= cc, lags= lags, samplerate= samplerate, timestamp= timestamp)
    return tau, shift

def onetapest(sigdict: list, which: int, diameter = 0.3):
    sig1 = sigdict[0].get("value1")
    sig2 = sigdict[1].get("value2")
    sig3 = sigdict[2].get("value3")
    sig4 = sigdict[3].get("value4")
    sig5 = sigdict[4].get("value5")
    sig6 = sigdict[5].get("value6")
    sig7 = sigdict[6].get("value7")
    sig8 = sigdict[7].get("value8")
    timestamp = sigdict[8].get("timestamp")
    
    avgtimestamp = findtimestampavg(timestamp)
    samplerate = int(1/(avgtimestamp*1e-9))
    
    radius = diameter/2
    ab = radius * 0.76536686473
    ac = radius * 1.41421356237
    ad = radius * 1.84775906502
    ae = float(diameter)
    
    match which:
        case 1:
            tof12=taufromsig(siga=sig1,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo12=safe_divide(ab,tof12)
            tof13=taufromsig(siga=sig1,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo13=safe_divide(ac,tof13)
            tof14=taufromsig(siga=sig1,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo14=safe_divide(ad,tof14)
            tof15=taufromsig(siga=sig1,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo15=safe_divide(ae,tof15)
            tof16=taufromsig(siga=sig1,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo16=safe_divide(ad,tof16)
            tof17=taufromsig(siga=sig1,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo17=safe_divide(ac,tof17)
            tof18=taufromsig(siga=sig1,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo18=safe_divide(ab,tof18)
            return np.array((0,velo12,velo13,velo14,velo15,velo16,velo17,velo18),dtype=np.float32)
        case 2:
            tof21=taufromsig(siga=sig2,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo21=safe_divide(ab,tof21)
            tof23=taufromsig(siga=sig2,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo23=safe_divide(ab,tof23)
            tof24=taufromsig(siga=sig2,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo24=safe_divide(ac,tof24)
            tof25=taufromsig(siga=sig2,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo25=safe_divide(ad,tof25)
            tof26=taufromsig(siga=sig2,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo26=safe_divide(ae,tof26)
            tof27=taufromsig(siga=sig2,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo27=safe_divide(ad,tof27)
            tof28=taufromsig(siga=sig2,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo28=safe_divide(ac,tof28)
            return np.array((velo21,0,velo23,velo24,velo25,velo26,velo27,velo28),dtype=np.float32)
        case 3:
            tof31=taufromsig(siga=sig3,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo31=safe_divide(ac,tof31)
            tof32=taufromsig(siga=sig3,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo32=safe_divide(ab,tof32)
            tof34=taufromsig(siga=sig3,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo34=safe_divide(ab,tof34)
            tof35=taufromsig(siga=sig3,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo35=safe_divide(ac,tof35)
            tof36=taufromsig(siga=sig3,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo36=safe_divide(ad,tof36)
            tof37=taufromsig(siga=sig3,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo37=safe_divide(ae,tof37)
            tof38=taufromsig(siga=sig3,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo38=safe_divide(ad,tof38)
            return np.array((velo31,velo32,0,velo34,velo35,velo36,velo37,velo38),dtype=np.float32)
        case 4:
            tof41=taufromsig(siga=sig4,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo41=safe_divide(ad,tof41)
            tof42=taufromsig(siga=sig4,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo42=safe_divide(ac,tof42)
            tof43=taufromsig(siga=sig4,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo43=safe_divide(ab,tof43)
            tof45=taufromsig(siga=sig4,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo45=safe_divide(ab,tof45)
            tof46=taufromsig(siga=sig4,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo46=safe_divide(ac,tof46)
            tof47=taufromsig(siga=sig4,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo47=safe_divide(ad,tof47)
            tof48=taufromsig(siga=sig4,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo48=safe_divide(ae,tof48)
            return np.array((velo41,velo42,velo43,0,velo45,velo46,velo47,velo48),dtype=np.float32)
        case 5:
            tof51=taufromsig(siga=sig5,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo51=safe_divide(ae,tof51)
            tof52=taufromsig(siga=sig5,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo52=safe_divide(ad,tof52)
            tof53=taufromsig(siga=sig5,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo53=safe_divide(ac,tof53)
            tof54=taufromsig(siga=sig5,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo54=safe_divide(ab,tof54)
            tof56=taufromsig(siga=sig5,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo56=safe_divide(ab,tof56)
            tof57=taufromsig(siga=sig5,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo57=safe_divide(ac,tof57)
            tof58=taufromsig(siga=sig5,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo58=safe_divide(ad,tof58)
            return np.array((velo51,velo52,velo53,velo54,0,velo56,velo57,velo58),dtype=np.float32)
        case 6:
            tof61=taufromsig(siga=sig6,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo61=safe_divide(ad,tof61)
            tof62=taufromsig(siga=sig6,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo62=safe_divide(ae,tof62)
            tof63=taufromsig(siga=sig6,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo63=safe_divide(ad,tof63)
            tof64=taufromsig(siga=sig6,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo64=safe_divide(ac,tof64)
            tof65=taufromsig(siga=sig6,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo65=safe_divide(ab,tof65)
            tof67=taufromsig(siga=sig6,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo67=safe_divide(ab,tof67)
            tof68=taufromsig(siga=sig6,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo68=safe_divide(ac,tof68)
            return np.array((velo61,velo62,velo63,velo64,velo65,0,velo67,velo68),dtype=np.float32)
        case 7:
            tof71=taufromsig(siga=sig7,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo71=safe_divide(ac,tof71)
            tof72=taufromsig(siga=sig7,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo72=safe_divide(ad,tof72)
            tof73=taufromsig(siga=sig7,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo73=safe_divide(ae,tof73)
            tof74=taufromsig(siga=sig7,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo74=safe_divide(ad,tof74)
            tof75=taufromsig(siga=sig7,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo75=safe_divide(ac,tof75)
            tof76=taufromsig(siga=sig7,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo76=safe_divide(ab,tof76)
            tof78=taufromsig(siga=sig7,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo78=safe_divide(ab,tof78)
            return np.array((velo71,velo72,velo73,velo74,velo75,velo76,0,velo78),dtype=np.float32)
        case 8:
            tof81=taufromsig(siga=sig8,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo81=safe_divide(ab,tof81)
            tof82=taufromsig(siga=sig8,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo82=safe_divide(ac,tof82)
            tof83=taufromsig(siga=sig8,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo83=safe_divide(ad,tof83)
            tof84=taufromsig(siga=sig8,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo84=safe_divide(ae,tof84)
            tof85=taufromsig(siga=sig8,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo85=safe_divide(ad,tof85)
            tof86=taufromsig(siga=sig8,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo86=safe_divide(ac,tof86)
            tof87=taufromsig(siga=sig8,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo87=safe_divide(ab,tof87)
            return np.array((velo81,velo82,velo83,velo84,velo85,velo86,velo87,0),dtype=np.float32)
        case _:
            raise ValueError

# ==============================================================================
# FUNGSI PROSES UTAMA (TELAH DIMODIFIKASI)
# ==============================================================================
def process_folder(folder_path, channel):
    print(f"\n[+] Memproses folder input: '{folder_path}'")

    # 1. Generate UUID
    # guid_survey = str(uuid.uuid4())
    guid_survey = f"SURVEY-{uuid.uuid4()}-{datetime.now().year}"
    print(f"[*] Menghasilkan GUID Survey baru: {guid_survey}")
    
    # 2. Verifikasi & Baca file input
    if not os.path.isdir(folder_path):
        print(f"[!] Error: Folder '{folder_path}' tidak ditemukan.")
        return
    try:
        json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
        if len(json_files) != 8:
            print(f"[!] Error: Ditemukan {len(json_files)} file JSON, seharusnya ada 8.")
            return
    except FileNotFoundError:
        print(f"[!] Error: Tidak dapat mengakses folder '{folder_path}'.")
        return

    file_map = {}
    for filename in json_files:
        match = re.search(r'(\d+)\.json$', filename, re.IGNORECASE)
        if match:
            file_map[int(match.group(1))] = os.path.join(folder_path, filename)
    
    if len(file_map) != 8:
        print(f"[!] Error: Tidak dapat menemukan 8 file JSON dengan nomor 1-8.")
        return

    # Urutkan file berdasarkan nomornya
    sorted_keys = sorted(file_map.keys())

    # 3. (BARU) Unggah semua file asli ke FTP dengan nama baru
    print("\n[*] Memulai proses unggah file-file asli ke FTP...")
    for i in sorted_keys:
        local_original_path = file_map[i]
        # Buat nama file baru sesuai format {guidsurvey}{nmrketukan}.json
        new_remote_filename = f"{guid_survey}{i}.json"
        upload_to_ftp(local_original_path, new_remote_filename)
    print("[*] Selesai mengunggah file asli.")

    # 4. Muat data dan jalankan perhitungan
    print("\n[*] Memuat data dari file JSON...")
    ketuk_data = [loadjson(file_map[i]) for i in sorted_keys]
    if any(data is None for data in ketuk_data): 
        print("[!] Gagal memuat satu atau lebih file JSON. Proses dibatalkan.")
        return

    print("\n[*] Menjalankan kalkulasi kecepatan (logika tidak diubah)...")
    veloketuk1=onetapest(ketuk_data[0],1,0.3); veloketuk2=onetapest(ketuk_data[1],2,0.3)
    veloketuk3=onetapest(ketuk_data[2],3,0.3); veloketuk4=onetapest(ketuk_data[3],4,0.3)
    veloketuk5=onetapest(ketuk_data[4],5,0.3); veloketuk6=onetapest(ketuk_data[5],6,0.3)
    veloketuk7=onetapest(ketuk_data[6],7,0.3); veloketuk8=onetapest(ketuk_data[7],8,0.3)
    matrix_list = np.vstack((veloketuk1,veloketuk2,veloketuk3,veloketuk4,veloketuk5,veloketuk6,veloketuk7,veloketuk8)).tolist()
    print("[*] Kalkulasi kecepatan selesai.")

    # 5. (MODIFIKASI) Susun data output dengan fileRow baru
    # Buat list nama file baru untuk field 'fileRow'
    file_row_list_baru = [f"{guid_survey}{i}.json" for i in sorted_keys]
    
    output_data = {
        "filename": f"{guid_survey}.json",
        "guid_survey": guid_survey,
        "matrix": matrix_list,
        "fileRow": file_row_list_baru # Menggunakan list nama file yang baru
    }

    # 6. Publikasikan hasil ke RabbitMQ
    publish_result_to_rmq(channel, RABBITMQ_PUBLISH_QUEUE, output_data)

    # 7. Simpan file hasil ke lokal
    local_save_dir = "hasil_json"
    os.makedirs(local_save_dir, exist_ok=True)
    output_filename = f"{guid_survey}.json"
    local_filepath = os.path.join(local_save_dir, output_filename)
    print(f"[*] Menyimpan hasil ke lokal: '{local_filepath}'")
    try:
        with codecs.open(local_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, separators=(',', ':'), sort_keys=False, indent=4)
        print(f"[✔] Berhasil menyimpan file lokal.")
    except Exception as e:
        print(f"[!] Error saat menyimpan file lokal: {e}")
        return

    # 8. Unggah file HASIL AKHIR ke FTP
    print(f"\n[*] Mengunggah file hasil akhir ({output_filename}) ke FTP...")
    upload_to_ftp(local_filepath, output_filename)
    
    print(f"--- Pemrosesan untuk '{folder_path}' selesai ---")

# ==============================================================================
# KONSUMEN RABBITMQ
# ==============================================================================
def main():
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST, port=RABBITMQ_PORT, virtual_host=RABBITMQ_VHOST,
        credentials=credentials, heartbeat=600, blocked_connection_timeout=300
    )
    while True:
        try:
            print("Mencoba terhubung ke RabbitMQ...")
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue=RABBITMQ_CONSUME_QUEUE, durable=True)
            print(f"[*] Berhasil terhubung. Menunggu pesan di antrian '{RABBITMQ_CONSUME_QUEUE}'.")

            def callback(ch, method, properties, body):
                folder_name = body.decode()
                print(f"\n[✔] Menerima pesan: '{folder_name}'")
                try:
                    # Teruskan 'channel' ke fungsi proses agar bisa dipakai untuk publish
                    process_folder(folder_name, ch)
                    ch.basic_ack(delivery_tag=method.delivery_tag) 
                except Exception as e:
                    print(f"[!!!] Terjadi error tak terduga saat memproses: {e}")
            
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=RABBITMQ_CONSUME_QUEUE, on_message_callback=callback)
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as e:
            print(f"Koneksi RabbitMQ gagal: {e}. Mencoba lagi dalam 5 detik...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nProses dihentikan.")
            sys.exit(0)
        except Exception as e:
            print(f"Error tak terduga: {e}. Mencoba menghubungkan ulang...")
            time.sleep(5)

if __name__ == '__main__':
    main()



# import numpy as np
# import json
# import codecs
# from gccestimating import GCC, corrlags
# import matplotlib.pyplot as plt
# import pika
# import time
# import os
# import sys
# import re
# import uuid
# from ftplib import FTP
# from dotenv import load_dotenv

# # ==============================================================================
# # KONFIGURASI DAN VARIABEL LINGKUNGAN
# # ==============================================================================

# load_dotenv()

# # --- KONFIGURASI RABBITMQ ---
# # RABBITMQ_HOST = 'rmq20.pptik.id'
# # RABBITMQ_PORT = 5672
# # RABBITMQ_USERNAME = 'terawang'
# # RABBITMQ_PASSWORD = 'Terawang@#2025'
# # RABBITMQ_VHOST = '/terawang'
# RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
# RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT"))
# RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
# RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
# RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST")
# RABBITMQ_CONSUME_QUEUE = 'terawang-folder' # Queue untuk menerima folder
# RABBITMQ_PUBLISH_QUEUE = 'terawangDataTof' # Queue baru untuk mengirim hasil

# # --- KONFIGURASI FTP (dari file .env) ---
# FTP_HOST = os.getenv("FTP_HOST")
# FTP_PORT = int(os.getenv("FTP_PORT", 21))
# FTP_USER = os.getenv("FTP_USER")
# FTP_PASSWORD = os.getenv("FTP_PASSWORD")
# FTP_RESULT_FOLDER = os.getenv("FTP_RESULT_FOLDER")

# # ==============================================================================
# # FUNGSI-FUNGSI BANTUAN (FTP & RMQ PUBLISH)
# # ==============================================================================

# def upload_to_ftp(local_filepath, remote_filename):
#     """Mengunggah file dari path lokal ke server FTP."""
#     if not all([FTP_HOST, FTP_USER, FTP_PASSWORD, FTP_RESULT_FOLDER]):
#         print("[!] Peringatan: Konfigurasi FTP tidak lengkap. Proses unggah dilewati.")
#         return False
        
#     try:
#         with FTP() as ftp:
#             print(f"  -> Menghubungkan ke FTP di {FTP_HOST}...")
#             ftp.connect(FTP_HOST, FTP_PORT)
#             ftp.login(FTP_USER, FTP_PASSWORD)
#             ftp.cwd(FTP_RESULT_FOLDER)
            
#             with open(local_filepath, 'rb') as file:
#                 print(f"  -> Mengunggah {remote_filename}...")
#                 ftp.storbinary(f'STOR {remote_filename}', file)
            
#             print(f"  -> Unggah FTP berhasil.")
#             return True
#     except Exception as e:
#         print(f"[!] Error saat mengunggah ke FTP: {e}")
#         return False

# def publish_result_to_rmq(channel, queue_name, json_data):
#     """Mempublikasikan data JSON ke antrian RabbitMQ yang ditentukan."""
#     try:
#         # Pastikan antrian tujuan ada (durable=True agar tidak hilang saat restart)
#         channel.queue_declare(queue=queue_name, durable=True)
        
#         # Konversi dictionary Python ke string JSON
#         message_body = json.dumps(json_data, indent=4)
        
#         # Publikasikan pesan
#         channel.basic_publish(
#             exchange='',
#             routing_key=queue_name,
#             body=message_body,
#             properties=pika.BasicProperties(
#                 delivery_mode=2,  # Membuat pesan persistent
#             )
#         )
#         print(f"[*] Berhasil mempublikasikan hasil ke antrian RMQ: '{queue_name}'")
#         return True
#     except Exception as e:
#         print(f"[!] Gagal mempublikasikan hasil ke RMQ: {e}")
#         return False
        
# # ==============================================================================
# # SEMUA FUNGSI PERHITUNGAN INTI (TIDAK DIUBAH)
# # ==============================================================================
# def loadcsv(filename, delim=","):
#     data = np.loadtxt(filename, delimiter=delim, dtype= np.float64)
#     return data

# def loadjson(filename):
#     print(f"  -> Membaca file: {os.path.basename(filename)}")
#     try:
#         with open(filename, 'r') as f:
#             data = json.load(f)
#         return data
#     except FileNotFoundError:
#         print(f"  -> ERROR: File tidak ditemukan di {filename}")
#         return None
#     except json.JSONDecodeError:
#         print(f"  -> ERROR: Format JSON tidak valid di {filename}")
#         return None

# def gccest(siga, sigb, samplerate=1, cctype="scot"):
#     n = int((len(siga)+ len(sigb)) / 2)
#     siga -= np.mean(siga, axis=0)
#     sigb -= np.mean(sigb, axis=0)
#     lags = corrlags(2*n-1, samplerate=samplerate)
#     gcc = GCC(sig1=siga,sig2=sigb)
    
#     match cctype.lower():
#         case "cc": cc = gcc.cc()
#         case "phat": cc = gcc.phat()
#         case "scot": cc = gcc.scot()
#         case "roth": cc = gcc.roth()
#         case "ht": cc = gcc.ht()
#         case _: cc = gcc.cc()
    
#     cc /= np.max(np.abs(cc))
#     return cc, lags

# def findtimestampavg(arr):
#     return np.mean(np.diff(arr))

# def timestampextrapfromavg(x,origin,arr):
#     return findtimestampavg(arr)*x + origin

# def tauest(cc, lags, samplerate = 1, timestamp = None):
#     shift = np.argmax(np.abs(cc))
#     tau = lags[shift] / float(samplerate)
#     if timestamp is not None:
#         if len(timestamp) < 999:
#             for x in range(1,251,1):
#                 timestamp = np.append(timestamp, timestampextrapfromavg(x=x,origin=timestamp[0], arr=timestamp))
#             for x in range(-1,-250,-1):
#                 timestamp = np.insert(timestamp, 0, timestampextrapfromavg(x=x,origin=timestamp[0], arr=timestamp))
#         peaktimestamp = timestamp[shift]
#         origintimestamp = timestamp[np.argmin(np.abs(lags))]
#         tau = peaktimestamp - origintimestamp
#         tau /= 1000000000
#     return np.abs(tau), shift

# def safe_divide(num, denom, default_value=0):
#     if not isinstance(num, (int, float)) or not isinstance(denom, (int, float)):
#         raise ValueError("Both num and denom must be numbers!")
#     else:
#         if denom == 0:
#             return default_value
#         else:
#             return num / denom

# def taufromsig(siga, sigb, samplerate= 1, timestamp= None):
#     cc, lags = gccest(siga= siga, sigb= sigb, samplerate= samplerate)
#     tau, shift = tauest(cc= cc, lags= lags, samplerate= samplerate, timestamp= timestamp)
#     return tau, shift

# def onetapest(sigdict: list, which: int, diameter = 0.3):
#     sig1 = sigdict[0].get("value1")
#     sig2 = sigdict[1].get("value2")
#     sig3 = sigdict[2].get("value3")
#     sig4 = sigdict[3].get("value4")
#     sig5 = sigdict[4].get("value5")
#     sig6 = sigdict[5].get("value6")
#     sig7 = sigdict[6].get("value7")
#     sig8 = sigdict[7].get("value8")
#     timestamp = sigdict[8].get("timestamp")
    
#     avgtimestamp = findtimestampavg(timestamp)
#     samplerate = int(1/(avgtimestamp*1e-9))
    
#     radius = diameter/2
#     ab = radius * 0.76536686473
#     ac = radius * 1.41421356237
#     ad = radius * 1.84775906502
#     ae = float(diameter)
    
#     match which:
#         case 1:
#             tof12=taufromsig(siga=sig1,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo12=safe_divide(ab,tof12)
#             tof13=taufromsig(siga=sig1,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo13=safe_divide(ac,tof13)
#             tof14=taufromsig(siga=sig1,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo14=safe_divide(ad,tof14)
#             tof15=taufromsig(siga=sig1,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo15=safe_divide(ae,tof15)
#             tof16=taufromsig(siga=sig1,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo16=safe_divide(ad,tof16)
#             tof17=taufromsig(siga=sig1,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo17=safe_divide(ac,tof17)
#             tof18=taufromsig(siga=sig1,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo18=safe_divide(ab,tof18)
#             return np.array((0,velo12,velo13,velo14,velo15,velo16,velo17,velo18),dtype=np.float32)
#         # ... (case 2 sampai 8 sama seperti sebelumnya) ...
#         case 2:
#             tof21=taufromsig(siga=sig2,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo21=safe_divide(ab,tof21)
#             tof23=taufromsig(siga=sig2,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo23=safe_divide(ab,tof23)
#             tof24=taufromsig(siga=sig2,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo24=safe_divide(ac,tof24)
#             tof25=taufromsig(siga=sig2,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo25=safe_divide(ad,tof25)
#             tof26=taufromsig(siga=sig2,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo26=safe_divide(ae,tof26)
#             tof27=taufromsig(siga=sig2,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo27=safe_divide(ad,tof27)
#             tof28=taufromsig(siga=sig2,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo28=safe_divide(ac,tof28)
#             return np.array((velo21,0,velo23,velo24,velo25,velo26,velo27,velo28),dtype=np.float32)
#         case 3:
#             tof31=taufromsig(siga=sig3,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo31=safe_divide(ac,tof31)
#             tof32=taufromsig(siga=sig3,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo32=safe_divide(ab,tof32)
#             tof34=taufromsig(siga=sig3,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo34=safe_divide(ab,tof34)
#             tof35=taufromsig(siga=sig3,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo35=safe_divide(ac,tof35)
#             tof36=taufromsig(siga=sig3,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo36=safe_divide(ad,tof36)
#             tof37=taufromsig(siga=sig3,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo37=safe_divide(ae,tof37)
#             tof38=taufromsig(siga=sig3,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo38=safe_divide(ad,tof38)
#             return np.array((velo31,velo32,0,velo34,velo35,velo36,velo37,velo38),dtype=np.float32)
#         case 4:
#             tof41=taufromsig(siga=sig4,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo41=safe_divide(ad,tof41)
#             tof42=taufromsig(siga=sig4,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo42=safe_divide(ac,tof42)
#             tof43=taufromsig(siga=sig4,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo43=safe_divide(ab,tof43)
#             tof45=taufromsig(siga=sig4,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo45=safe_divide(ab,tof45)
#             tof46=taufromsig(siga=sig4,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo46=safe_divide(ac,tof46)
#             tof47=taufromsig(siga=sig4,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo47=safe_divide(ad,tof47)
#             tof48=taufromsig(siga=sig4,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo48=safe_divide(ae,tof48)
#             return np.array((velo41,velo42,velo43,0,velo45,velo46,velo47,velo48),dtype=np.float32)
#         case 5:
#             tof51=taufromsig(siga=sig5,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo51=safe_divide(ae,tof51)
#             tof52=taufromsig(siga=sig5,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo52=safe_divide(ad,tof52)
#             tof53=taufromsig(siga=sig5,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo53=safe_divide(ac,tof53)
#             tof54=taufromsig(siga=sig5,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo54=safe_divide(ab,tof54)
#             tof56=taufromsig(siga=sig5,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo56=safe_divide(ab,tof56)
#             tof57=taufromsig(siga=sig5,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo57=safe_divide(ac,tof57)
#             tof58=taufromsig(siga=sig5,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo58=safe_divide(ad,tof58)
#             return np.array((velo51,velo52,velo53,velo54,0,velo56,velo57,velo58),dtype=np.float32)
#         case 6:
#             tof61=taufromsig(siga=sig6,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo61=safe_divide(ad,tof61)
#             tof62=taufromsig(siga=sig6,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo62=safe_divide(ae,tof62)
#             tof63=taufromsig(siga=sig6,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo63=safe_divide(ad,tof63)
#             tof64=taufromsig(siga=sig6,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo64=safe_divide(ac,tof64)
#             tof65=taufromsig(siga=sig6,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo65=safe_divide(ab,tof65)
#             tof67=taufromsig(siga=sig6,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo67=safe_divide(ab,tof67)
#             tof68=taufromsig(siga=sig6,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo68=safe_divide(ac,tof68)
#             return np.array((velo61,velo62,velo63,velo64,velo65,0,velo67,velo68),dtype=np.float32)
#         case 7:
#             tof71=taufromsig(siga=sig7,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo71=safe_divide(ac,tof71)
#             tof72=taufromsig(siga=sig7,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo72=safe_divide(ad,tof72)
#             tof73=taufromsig(siga=sig7,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo73=safe_divide(ae,tof73)
#             tof74=taufromsig(siga=sig7,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo74=safe_divide(ad,tof74)
#             tof75=taufromsig(siga=sig7,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo75=safe_divide(ac,tof75)
#             tof76=taufromsig(siga=sig7,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo76=safe_divide(ab,tof76)
#             tof78=taufromsig(siga=sig7,sigb=sig8,samplerate=samplerate,timestamp=timestamp)[0];velo78=safe_divide(ab,tof78)
#             return np.array((velo71,velo72,velo73,velo74,velo75,velo76,0,velo78),dtype=np.float32)
#         case 8:
#             tof81=taufromsig(siga=sig8,sigb=sig1,samplerate=samplerate,timestamp=timestamp)[0];velo81=safe_divide(ab,tof81)
#             tof82=taufromsig(siga=sig8,sigb=sig2,samplerate=samplerate,timestamp=timestamp)[0];velo82=safe_divide(ac,tof82)
#             tof83=taufromsig(siga=sig8,sigb=sig3,samplerate=samplerate,timestamp=timestamp)[0];velo83=safe_divide(ad,tof83)
#             tof84=taufromsig(siga=sig8,sigb=sig4,samplerate=samplerate,timestamp=timestamp)[0];velo84=safe_divide(ae,tof84)
#             tof85=taufromsig(siga=sig8,sigb=sig5,samplerate=samplerate,timestamp=timestamp)[0];velo85=safe_divide(ad,tof85)
#             tof86=taufromsig(siga=sig8,sigb=sig6,samplerate=samplerate,timestamp=timestamp)[0];velo86=safe_divide(ac,tof86)
#             tof87=taufromsig(siga=sig8,sigb=sig7,samplerate=samplerate,timestamp=timestamp)[0];velo87=safe_divide(ab,tof87)
#             return np.array((velo81,velo82,velo83,velo84,velo85,velo86,velo87,0),dtype=np.float32)
#         case _:
#             raise ValueError
# # ==============================================================================
# # FUNGSI PROSES UTAMA
# # ==============================================================================
# def process_folder(folder_path, channel):
#     print(f"\n[+] Memproses folder input: '{folder_path}'")

#     # 1. Generate UUID
#     guid_survey = str(uuid.uuid4())
#     print(f"[*] Menghasilkan GUID Survey baru: {guid_survey}")
    
#     # 2. Verifikasi & Baca file input
#     if not os.path.isdir(folder_path):
#         print(f"[!] Error: Folder '{folder_path}' tidak ditemukan.")
#         return
#     try:
#         json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
#         if len(json_files) != 8:
#             print(f"[!] Error: Ditemukan {len(json_files)} file JSON, seharusnya ada 8.")
#             return
#     except FileNotFoundError:
#         print(f"[!] Error: Tidak dapat mengakses folder '{folder_path}'.")
#         return

#     file_map = {}
#     for filename in json_files:
#         match = re.search(r'(\d+)\.json$', filename, re.IGNORECASE)
#         if match:
#             file_map[int(match.group(1))] = os.path.join(folder_path, filename)
    
#     if len(file_map) != 8:
#         print(f"[!] Error: Tidak dapat menemukan 8 file JSON dengan nomor 1-8.")
#         return

#     # 3. Muat data dan jalankan perhitungan
#     file_row_list = [os.path.basename(file_map[i]) for i in sorted(file_map.keys())]
#     ketuk_data = [loadjson(file_map[i]) for i in sorted(file_map.keys())]
#     if any(data is None for data in ketuk_data): return

#     print("[*] Menjalankan kalkulasi kecepatan...")
#     veloketuk1=onetapest(ketuk_data[0],1,0.3); veloketuk2=onetapest(ketuk_data[1],2,0.3)
#     veloketuk3=onetapest(ketuk_data[2],3,0.3); veloketuk4=onetapest(ketuk_data[3],4,0.3)
#     veloketuk5=onetapest(ketuk_data[4],5,0.3); veloketuk6=onetapest(ketuk_data[5],6,0.3)
#     veloketuk7=onetapest(ketuk_data[6],7,0.3); veloketuk8=onetapest(ketuk_data[7],8,0.3)
#     matrix_list = np.vstack((veloketuk1,veloketuk2,veloketuk3,veloketuk4,veloketuk5,veloketuk6,veloketuk7,veloketuk8)).tolist()

#     # 4. Susun data output
#     output_data = {
#         "filename": f"{guid_survey}.json",
#         "guid_survey": guid_survey,
#         "matrix": matrix_list,
#         "fileRow": file_row_list
#     }

#     # 5. Publikasikan hasil ke RabbitMQ
#     publish_result_to_rmq(channel, RABBITMQ_PUBLISH_QUEUE, output_data)

#     # 6. Simpan hasil ke file lokal
#     local_save_dir = "hasil_json"
#     os.makedirs(local_save_dir, exist_ok=True)
#     output_filename = f"{guid_survey}.json"
#     local_filepath = os.path.join(local_save_dir, output_filename)
#     print(f"[*] Menyimpan hasil ke lokal: '{local_filepath}'")
#     try:
#         with codecs.open(local_filepath, 'w', encoding='utf-8') as f:
#             json.dump(output_data, f, separators=(',', ':'), sort_keys=False, indent=4)
#         print(f"[✔] Berhasil menyimpan file lokal.")
#     except Exception as e:
#         print(f"[!] Error saat menyimpan file lokal: {e}")
#         return

#     # 7. Unggah file ke FTP
#     upload_to_ftp(local_filepath, output_filename)
    
#     print(f"--- Pemrosesan untuk '{folder_path}' selesai ---")

# # ==============================================================================
# # KONSUMEN RABBITMQ
# # ==============================================================================
# def main():
#     credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
#     parameters = pika.ConnectionParameters(
#         host=RABBITMQ_HOST, port=RABBITMQ_PORT, virtual_host=RABBITMQ_VHOST,
#         credentials=credentials, heartbeat=600, blocked_connection_timeout=300
#     )
#     while True:
#         try:
#             print("Mencoba terhubung ke RabbitMQ...")
#             connection = pika.BlockingConnection(parameters)
#             channel = connection.channel()
#             channel.queue_declare(queue=RABBITMQ_CONSUME_QUEUE, durable=True)
#             print(f"[*] Berhasil terhubung. Menunggu pesan di antrian '{RABBITMQ_CONSUME_QUEUE}'.")

#             def callback(ch, method, properties, body):
#                 folder_name = body.decode()
#                 print(f"\n[✔] Menerima pesan: '{folder_name}'")
#                 try:
#                     # Teruskan 'channel' ke fungsi proses agar bisa dipakai untuk publish
#                     process_folder(folder_name, ch)
#                     ch.basic_ack(delivery_tag=method.delivery_tag) 
#                 except Exception as e:
#                     print(f"[!!!] Terjadi error tak terduga saat memproses: {e}")
            
#             channel.basic_qos(prefetch_count=1)
#             channel.basic_consume(queue=RABBITMQ_CONSUME_QUEUE, on_message_callback=callback)
#             channel.start_consuming()

#         except pika.exceptions.AMQPConnectionError as e:
#             print(f"Koneksi RabbitMQ gagal: {e}. Mencoba lagi dalam 5 detik...")
#             time.sleep(5)
#         except KeyboardInterrupt:
#             print("\nProses dihentikan.")
#             sys.exit(0)
#         except Exception as e:
#             print(f"Error tak terduga: {e}. Mencoba menghubungkan ulang...")
#             time.sleep(5)

# if __name__ == '__main__':
#     main()