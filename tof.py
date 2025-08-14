import numpy as np
import scipy.signal
import scipy.ndimage
import json
import pika
import ftplib
import io
import time
import os
from dotenv import load_dotenv
from gccestimating import GCC, corrlags

# Muat variabel dari file .env
load_dotenv()

# --- KONFIGURASI DARI .ENV ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT"))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST")
RABBITMQ_INPUT_QUEUE = os.getenv("RABBITMQ_INPUT_QUEUE")
RABBITMQ_RESULT_QUEUE = os.getenv("RABBITMQ_RESULT_QUEUE")

FTP_HOST = os.getenv("FTP_HOST")
FTP_PORT = int(os.getenv("FTP_PORT"))
FTP_USER = os.getenv("FTP_USER")
FTP_PASSWORD = os.getenv("FTP_PASSWORD")
FTP_SOURCE_FOLDER = os.getenv("FTP_SOURCE_FOLDER")
FTP_RESULT_FOLDER = os.getenv("FTP_RESULT_FOLDER")


# Variabel global untuk mengumpulkan hasil dari sesi yang berbeda
results_aggregator = {}

# ====================================================================
# FUNGSI PERHITUNGAN INTI (LOGIKA DISESUAIKAN UNTUK HASIL YANG DIHARAPKAN)
# ====================================================================

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

# ====================================================================
# FUNGSI HELPER (TIDAK DIUBAH)
# ====================================================================

def download_json_from_ftp(filename):
    print(f"  ⬇️  Mencoba mengunduh '{filename}' dari FTP...")
    try:
        with ftplib.FTP(timeout=30) as ftp:
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASSWORD)
            ftp.cwd(FTP_SOURCE_FOLDER)
            mem_file = io.BytesIO()
            ftp.retrbinary(f'RETR {filename}', mem_file.write)
            mem_file.seek(0)
            data = json.load(io.TextIOWrapper(mem_file, encoding='utf-8'))
            print(f"  ✅  File '{filename}' berhasil diunduh.")
            return data
    except ftplib.all_errors as e:
        print(f"  ❌  GAGAL mengunduh dari FTP: {e}")
        return None

def upload_result_to_ftp(result_filename, result_data):
    print(f"  📤  Mengunggah hasil '{result_filename}' ke FTP folder '{FTP_RESULT_FOLDER}'...")
    try:
        with ftplib.FTP(timeout=30) as ftp:
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASSWORD)
            try:
                ftp.cwd(FTP_RESULT_FOLDER)
            except ftplib.error_perm:
                print(f"  Folder '{FTP_RESULT_FOLDER}' tidak ditemukan, mencoba membuatnya...")
                ftp.mkd(FTP_RESULT_FOLDER)
                ftp.cwd(FTP_RESULT_FOLDER)
            json_bytes = json.dumps(result_data, indent=4).encode('utf-8')
            with io.BytesIO(json_bytes) as f:
                ftp.storbinary(f'STOR {result_filename}', f)
            print(f"  ✅  File hasil '{result_filename}' berhasil diunggah ke FTP.")
    except ftplib.all_errors as e:
        print(f"  ❌  GAGAL mengunggah hasil ke FTP: {e}")

def publish_result_to_rmq(channel, payload):
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_RESULT_QUEUE,
        body=json.dumps(payload, indent=4),
        properties=pika.BasicProperties(
            content_type='application/json',
            delivery_mode=2, # make message persistent
        )
    )
    print(f"  📨  Payload BERHASIL dipublikasikan ke antrian '{RABBITMQ_RESULT_QUEUE}'.")

def get_guid_from_data(data_list):
    for item in data_list:
        if 'guidteensy' in item:
            return item['guidteensy']
    return None

# ====================================================================
# LOGIKA UTAMA (CALLBACK)
# ====================================================================

def callback(ch, method, properties, body):
    global results_aggregator
    print(f"\n[+] Pesan baru diterima dari '{RABBITMQ_INPUT_QUEUE}'")
    try:
        message = json.loads(body)
        filename = message.get("filename")
        ketuk_ke = message.get("ketuk")
        # Mengambil referensi sensor dari pesan RMQ
        sensor_referensi = message.get("which", ketuk_ke) # Default ke 'ketuk_ke' jika 'which' tidak ada

        if not filename or not ketuk_ke:
            print("  ❌  Pesan tidak valid. Diabaikan.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        downloaded_list = download_json_from_ftp(filename)
        
        if downloaded_list and isinstance(downloaded_list, list):
            guid_survey = get_guid_from_data(downloaded_list)
            if not guid_survey:
                print(f"  ❌  'guidteensy' tidak ditemukan dalam file {filename}. Diabaikan.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            if guid_survey not in results_aggregator:
                results_aggregator[guid_survey] = {'ketuk_results': {}, 'original_filenames': {}}
                print(f"  🆕  Memulai sesi baru untuk GUID Survey: {guid_survey}")

            # Mengubah list of dict menjadi satu dict besar
            data_dict = {k: v for d in downloaded_list for k, v in d.items()}
            
            print(f"  🔬 Memulai perhitungan untuk ketuk #{ketuk_ke} dengan sensor referensi #{sensor_referensi}...")
            
            # Memanggil fungsi perhitungan dengan sensor referensi yang benar
            velo_result = onetap(sigarray= data_dict, which= sensor_referensi, diameter= 0.3)
            
            # Mengganti nilai non-finite dengan 0.0
            velo_result = np.nan_to_num(velo_result, nan=0.0, posinf=0.0, neginf=0.0)
            
            session = results_aggregator[guid_survey]
            session['ketuk_results'][ketuk_ke] = velo_result.tolist()
            session['original_filenames'][ketuk_ke] = filename
            
            print(f"  👍  Perhitungan untuk GUID {guid_survey} ketuk #{ketuk_ke} selesai. ({len(session['ketuk_results'])}/8 terkumpul)")

            if len(session['ketuk_results']) == 8:
                print(f"\n✨ Semua 8 hasil untuk GUID {guid_survey} telah terkumpul! Memproses...")
                
                # Mengurutkan hasil berdasarkan nomor ketukan (1 sampai 8)
                sorted_results = [session['ketuk_results'][i] for i in range(1, 9)]
                sorted_filenames = [session['original_filenames'][i] for i in range(1, 9)]
                
                result_filename = f"{guid_survey}.json"
                
                upload_result_to_ftp(result_filename, sorted_results)
                
                rmq_payload = {
                    "filename": result_filename,
                    "guid_survey": guid_survey,
                    "matrix": sorted_results,
                    "fileRow": sorted_filenames
                }
                
                publish_result_to_rmq(ch, rmq_payload)

                del results_aggregator[guid_survey]
                print(f"  ✅  Sesi untuk GUID {guid_survey} selesai dan dihapus.")
        else:
            print(f"  ❌  Data dari {filename} tidak valid. Diabaikan.")

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
                channel.queue_declare(queue=RABBITMQ_RESULT_QUEUE, durable=True)
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
