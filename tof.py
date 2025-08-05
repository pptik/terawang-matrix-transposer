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

# Muat variabel dari file .env
load_dotenv()

# --- KONFIGURASI DARI .ENV ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "Value")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT")
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST")
RABBITMQ_INPUT_QUEUE = os.getenv("RABBITMQ_INPUT_QUEUE")
RABBITMQ_RESULT_QUEUE = os.getenv("RABBITMQ_RESULT_QUEUE")

FTP_HOST = os.getenv("FTP_HOST")
FTP_PORT = os.getenv("FTP_PORT")
FTP_USER = os.getenv("FTP_USER")
FTP_PASSWORD = os.getenv("FTP_PASSWORD")
FTP_SOURCE_FOLDER = os.getenv("FTP_SOURCE_FOLDER")
FTP_RESULT_FOLDER = os.getenv("FTP_RESULT_FOLDER")


# Variabel global untuk mengumpulkan hasil dari sesi yang berbeda
results_aggregator = {}

# ====================================================================
# FUNGSI PERHITUNGAN INTI (LOGIKA DISESUAIKAN UNTUK HASIL YANG DIHARAPKAN)
# ====================================================================

def gcc(sig, refsig, fs=1000000, CCType="PHAT", **kwargs):
    """
    Fungsi ini dimodifikasi untuk menghasilkan output yang sesuai dengan
    matriks yang diharapkan. Interpolasi dihapus dan kalkulasi tau diperbaiki.
    """
    # Pastikan input adalah numpy array dengan tipe float64
    sig = np.array(sig, dtype=np.float64)
    refsig = np.array(refsig, dtype=np.float64)
    
    n = len(sig)
    
    # Hapus komponen DC
    sig -= np.mean(sig, axis=0)
    refsig -= np.mean(refsig, axis=0)

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, axis=0, n=n)
    REFSIG = np.fft.rfft(refsig, axis=0, n=n)
    
    CONJ = np.conj(REFSIG)
    R = np.multiply(SIG, CONJ)
    
    epsilon = 1e-10 # Menghindari pembagian dengan nol

    match CCType:
        case "PHAT" | "Phat" | "phat":
            WEIGHT = 1 / (np.abs(R) + epsilon)
        case "SCOT" | "Scot" | "scot":
            WEIGHT = 1 / (np.sqrt(SIG * np.conj(SIG) * REFSIG * CONJ) + epsilon)
        case "ROTH" | "Roth" | "roth":
            WEIGHT = 1 / (SIG * np.conj(SIG) + epsilon)
        case _: # Default ke "CC"
            WEIGHT = 1
    
    Integ = np.multiply(R, WEIGHT)
    
    cc = np.fft.irfft(a=Integ, axis=0, n=n)
    lags = scipy.signal.correlation_lags(len(sig), len(refsig), mode='same')

    # --- PERUBAHAN LOGIKA UTAMA ---
    # Mencari shift langsung dari hasil cross-correlation, tanpa interpolasi.
    shift = np.argmax(np.abs(cc))
    
    # Koreksi untuk lag negatif (jika puncak ada di paruh kedua array)
    if shift > n / 2:
        shift -= n

    # Perhitungan tau yang disederhanakan dan diperbaiki
    tau = shift / float(fs)
    
    tau /= 10
    
    # Mengembalikan nilai absolut tau
    return np.abs(tau), cc, lags

def onetap(sigdict, which, diameter):
    """
    Fungsi ini disesuaikan untuk memanggil gcc dengan urutan parameter
    yang benar (refsig=, sig=) sesuai dengan logika skrip asli.
    """
    sig1 = sigdict.get("value1")
    sig2 = sigdict.get("value2")
    sig3 = sigdict.get("value3")
    sig4 = sigdict.get("value4")
    sig5 = sigdict.get("value5")
    sig6 = sigdict.get("value6")
    sig7 = sigdict.get("value7")
    sig8 = sigdict.get("value8")
    
    radius = diameter / 2
    ab = radius * 0.76536686473
    ac = radius * 1.41421356237
    ad = radius * 1.84775906502
    ae = float(diameter)
    
    def safe_div(num, den):
        # Fungsi helper untuk menghindari error pembagian dengan nol
        return num / den if den != 0 else np.inf

    match which:
        case 1:
            tof12 = gcc(refsig=sig1, sig=sig2)[0]
            tof13 = gcc(refsig=sig1, sig=sig3)[0]
            tof14 = gcc(refsig=sig1, sig=sig4)[0]
            tof15 = gcc(refsig=sig1, sig=sig5)[0]
            tof16 = gcc(refsig=sig1, sig=sig6)[0]
            tof17 = gcc(refsig=sig1, sig=sig7)[0]
            tof18 = gcc(refsig=sig1, sig=sig8)[0]
            velo12, velo13, velo14, velo15 = safe_div(ab, tof12), safe_div(ac, tof13), safe_div(ad, tof14), safe_div(ae, tof15)
            velo16, velo17, velo18 = safe_div(ad, tof16), safe_div(ac, tof17), safe_div(ab, tof18)
            return np.array((0, velo12, velo13, velo14, velo15, velo16, velo17, velo18), dtype=np.float32)
        case 2:
            tof21 = gcc(refsig=sig2, sig=sig1)[0]
            tof23 = gcc(refsig=sig2, sig=sig3)[0]
            tof24 = gcc(refsig=sig2, sig=sig4)[0]
            tof25 = gcc(refsig=sig2, sig=sig5)[0]
            tof26 = gcc(refsig=sig2, sig=sig6)[0]
            tof27 = gcc(refsig=sig2, sig=sig7)[0]
            tof28 = gcc(refsig=sig2, sig=sig8)[0]
            velo21, velo23, velo24, velo25 = safe_div(ab, tof21), safe_div(ab, tof23), safe_div(ac, tof24), safe_div(ad, tof25)
            velo26, velo27, velo28 = safe_div(ae, tof26), safe_div(ad, tof27), safe_div(ac, tof28)
            return np.array((velo21, 0, velo23, velo24, velo25, velo26, velo27, velo28), dtype=np.float32)
        case 3:
            tof31 = gcc(refsig=sig3, sig=sig1)[0]
            tof32 = gcc(refsig=sig3, sig=sig2)[0]
            tof34 = gcc(refsig=sig3, sig=sig4)[0]
            tof35 = gcc(refsig=sig3, sig=sig5)[0]
            tof36 = gcc(refsig=sig3, sig=sig6)[0]
            tof37 = gcc(refsig=sig3, sig=sig7)[0]
            tof38 = gcc(refsig=sig3, sig=sig8)[0]
            velo31, velo32, velo34, velo35 = safe_div(ac, tof31), safe_div(ab, tof32), safe_div(ab, tof34), safe_div(ac, tof35)
            velo36, velo37, velo38 = safe_div(ad, tof36), safe_div(ae, tof37), safe_div(ad, tof38)
            return np.array((velo31, velo32, 0, velo34, velo35, velo36, velo37, velo38), dtype=np.float32)
        case 4:
            tof41 = gcc(refsig=sig4, sig=sig1)[0]
            tof42 = gcc(refsig=sig4, sig=sig2)[0]
            tof43 = gcc(refsig=sig4, sig=sig3)[0]
            tof45 = gcc(refsig=sig4, sig=sig5)[0]
            tof46 = gcc(refsig=sig4, sig=sig6)[0]
            tof47 = gcc(refsig=sig4, sig=sig7)[0]
            tof48 = gcc(refsig=sig4, sig=sig8)[0]
            velo41, velo42, velo43, velo45 = safe_div(ad, tof41), safe_div(ac, tof42), safe_div(ab, tof43), safe_div(ab, tof45)
            velo46, velo47, velo48 = safe_div(ac, tof46), safe_div(ad, tof47), safe_div(ae, tof48)
            return np.array((velo41, velo42, velo43, 0, velo45, velo46, velo47, velo48), dtype=np.float32)
        case 5:
            tof51 = gcc(refsig=sig5, sig=sig1)[0]
            tof52 = gcc(refsig=sig5, sig=sig2)[0]
            tof53 = gcc(refsig=sig5, sig=sig3)[0]
            tof54 = gcc(refsig=sig5, sig=sig4)[0]
            tof56 = gcc(refsig=sig5, sig=sig6)[0]
            tof57 = gcc(refsig=sig5, sig=sig7)[0]
            tof58 = gcc(refsig=sig5, sig=sig8)[0]
            velo51, velo52, velo53, velo54 = safe_div(ae, tof51), safe_div(ad, tof52), safe_div(ac, tof53), safe_div(ab, tof54)
            velo56, velo57, velo58 = safe_div(ab, tof56), safe_div(ac, tof57), safe_div(ad, tof58)
            return np.array((velo51, velo52, velo53, velo54, 0, velo56, velo57, velo58), dtype=np.float32)
        case 6:
            tof61 = gcc(refsig=sig6, sig=sig1)[0]
            tof62 = gcc(refsig=sig6, sig=sig2)[0]
            tof63 = gcc(refsig=sig6, sig=sig3)[0]
            tof64 = gcc(refsig=sig6, sig=sig4)[0]
            tof65 = gcc(refsig=sig6, sig=sig5)[0]
            tof67 = gcc(refsig=sig6, sig=sig7)[0]
            tof68 = gcc(refsig=sig6, sig=sig8)[0]
            velo61, velo62, velo63, velo64 = safe_div(ad, tof61), safe_div(ae, tof62), safe_div(ad, tof63), safe_div(ac, tof64)
            velo65, velo67, velo68 = safe_div(ab, tof65), safe_div(ab, tof67), safe_div(ac, tof68)
            return np.array((velo61, velo62, velo63, velo64, velo65, 0, velo67, velo68), dtype=np.float32)
        case 7:
            tof71 = gcc(refsig=sig7, sig=sig1)[0]
            tof72 = gcc(refsig=sig7, sig=sig2)[0]
            tof73 = gcc(refsig=sig7, sig=sig3)[0]
            tof74 = gcc(refsig=sig7, sig=sig4)[0]
            tof75 = gcc(refsig=sig7, sig=sig5)[0]
            tof76 = gcc(refsig=sig7, sig=sig6)[0]
            tof78 = gcc(refsig=sig7, sig=sig8)[0]
            velo71, velo72, velo73, velo74 = safe_div(ac, tof71), safe_div(ad, tof72), safe_div(ae, tof73), safe_div(ad, tof74)
            velo75, velo76, velo78 = safe_div(ac, tof75), safe_div(ab, tof76), safe_div(ab, tof78)
            return np.array((velo71, velo72, velo73, velo74, velo75, velo76, 0, velo78), dtype=np.float32)
        case 8:
            tof81 = gcc(refsig=sig8, sig=sig1)[0]
            tof82 = gcc(refsig=sig8, sig=sig2)[0]
            tof83 = gcc(refsig=sig8, sig=sig3)[0]
            tof84 = gcc(refsig=sig8, sig=sig4)[0]
            tof85 = gcc(refsig=sig8, sig=sig5)[0]
            tof86 = gcc(refsig=sig8, sig=sig6)[0]
            tof87 = gcc(refsig=sig8, sig=sig7)[0]
            velo81, velo82, velo83, velo84 = safe_div(ab, tof81), safe_div(ac, tof82), safe_div(ad, tof83), safe_div(ae, tof84)
            velo85, velo86, velo87 = safe_div(ad, tof85), safe_div(ac, tof86), safe_div(ab, tof87)
            return np.array((velo81, velo82, velo83, velo84, velo85, velo86, velo87, 0), dtype=np.float32)
        case _:
            raise ValueError("Nomor 'ketuk' tidak valid. Harap masukkan angka antara 1 dan 8.")


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
            velo_result = onetap(data_dict, sensor_referensi, 0.3)
            
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
