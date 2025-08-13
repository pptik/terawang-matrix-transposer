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

# Variabel global untuk mengumpulkan hasil
results_aggregator = {}

# ====================================================================
# BAGIAN PERHITUNGAN (REPLIKASI 1:1 DARI PERILAKU KODE ASLI)
# ====================================================================

def gccnormal(sig, refsig, fs=1000000, interp=128, max_tau=None, CCType="PHAT", timestamp=None):
    """
    FUNGSI INI KINI MEREPLIKASI PERILAKU ASLI DENGAN TEPAT,
    TERMASUK CARA MENANGANI KASUS EKSTREM.
    """
    sig = np.array(sig, dtype=np.float64)
    refsig = np.array(refsig, dtype=np.float64)
    n = len(sig)
    
    sig -= np.mean(sig, axis=0)
    refsig -= np.mean(refsig, axis=0)

    SIG = np.fft.rfft(sig, axis=0, n=n)
    REFSIG = np.fft.rfft(refsig, axis=0, n=n)
    
    CONJ = np.conj(SIG)
    R = np.multiply(REFSIG, CONJ)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        match CCType.upper():
            case "PHAT":
                WEIGHT = 1.0 / np.abs(R)
            case "SCOT":
                WEIGHT = 1.0 / np.sqrt((SIG * np.conj(SIG)) * (REFSIG * np.conj(REFSIG)))
            case "ROTH":
                WEIGHT = 1.0 / (SIG * np.conj(SIG))
            case _:
                WEIGHT = 1.0
    WEIGHT[np.isinf(WEIGHT) | np.isnan(WEIGHT)] = 0
    
    Integ = np.multiply(R, WEIGHT)
    cc = np.fft.irfft(a=Integ, axis=0, n=n)
    lags = scipy.signal.correlation_lags(len(refsig), len(sig), mode='same')

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    
    if max_shift * 2 + 1 > len(cc):
        max_shift = (len(cc) - 1) // 2
        
    smallcc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    # --- KUNCI REPLIKASI PERILAKU ---
    # Melakukan pembagian "tidak aman" seperti kode asli,
    # yang mungkin menghasilkan inf/nan, yang mana ini PENTING untuk hasil akhir.
    with np.errstate(divide='ignore', invalid='ignore'):
        smallcc /= np.max(cc)

    # Ganti nilai nan dengan -inf agar argmax mengabaikannya
    smallcc[np.isnan(smallcc)] = -np.inf
    
    shift = np.argmax(smallcc) - max_shift
    tau = shift / float(interp * fs)
    
    return np.abs(tau), cc, lags

def onetap(sigdict: list, which: int, diameter=0.3):
    """FUNGSI ASLI ANDA - HANYA MEMANGGIL FUNGSI gccnormal YANG SUDAH DIREPLIKASI"""
    sig1, sig2, sig3, sig4 = sigdict[0].get("value1"), sigdict[1].get("value2"), sigdict[2].get("value3"), sigdict[3].get("value4")
    sig5, sig6, sig7, sig8 = sigdict[4].get("value5"), sigdict[5].get("value6"), sigdict[6].get("value7"), sigdict[7].get("value8")

    soundspeed = 4150
    radius = diameter / 2
    ab, ac, ad, ae = radius * 0.76536686473, radius * 1.41421356237, radius * 1.84775906502, float(diameter)
    
    max_tau_val = diameter / soundspeed

    def safe_div(num, den):
        return num / den if den > 1e-12 else 0.0

    def get_tof(ref, sig):
        return gccnormal(refsig=ref, sig=sig, timestamp=None, max_tau=max_tau_val)[0]

    match which:
        case 1: return np.array((0, safe_div(ab, get_tof(sig1, sig2)), safe_div(ac, get_tof(sig1, sig3)), safe_div(ad, get_tof(sig1, sig4)), safe_div(ae, get_tof(sig1, sig5)), safe_div(ad, get_tof(sig1, sig6)), safe_div(ac, get_tof(sig1, sig7)), safe_div(ab, get_tof(sig1, sig8))), dtype=np.float32)
        case 2: return np.array((safe_div(ab, get_tof(sig2, sig1)), 0, safe_div(ab, get_tof(sig2, sig3)), safe_div(ac, get_tof(sig2, sig4)), safe_div(ad, get_tof(sig2, sig5)), safe_div(ae, get_tof(sig2, sig6)), safe_div(ad, get_tof(sig2, sig7)), safe_div(ac, get_tof(sig2, sig8))), dtype=np.float32)
        case 3: return np.array((safe_div(ac, get_tof(sig3, sig1)), safe_div(ab, get_tof(sig3, sig2)), 0, safe_div(ab, get_tof(sig3, sig4)), safe_div(ac, get_tof(sig3, sig5)), safe_div(ad, get_tof(sig3, sig6)), safe_div(ae, get_tof(sig3, sig7)), safe_div(ad, get_tof(sig3, sig8))), dtype=np.float32)
        case 4: return np.array((safe_div(ad, get_tof(sig4, sig1)), safe_div(ac, get_tof(sig4, sig2)), safe_div(ab, get_tof(sig4, sig3)), 0, safe_div(ab, get_tof(sig4, sig5)), safe_div(ac, get_tof(sig4, sig6)), safe_div(ad, get_tof(sig4, sig7)), safe_div(ae, get_tof(sig4, sig8))), dtype=np.float32)
        case 5: return np.array((safe_div(ae, get_tof(sig5, sig1)), safe_div(ad, get_tof(sig5, sig2)), safe_div(ac, get_tof(sig5, sig3)), safe_div(ab, get_tof(sig5, sig4)), 0, safe_div(ab, get_tof(sig5, sig6)), safe_div(ac, get_tof(sig5, sig7)), safe_div(ad, get_tof(sig5, sig8))), dtype=np.float32)
        case 6: return np.array((safe_div(ad, get_tof(sig6, sig1)), safe_div(ae, get_tof(sig6, sig2)), safe_div(ad, get_tof(sig6, sig3)), safe_div(ac, get_tof(sig6, sig4)), safe_div(ab, get_tof(sig6, sig5)), 0, safe_div(ab, get_tof(sig6, sig7)), safe_div(ac, get_tof(sig6, sig8))), dtype=np.float32)
        case 7: return np.array((safe_div(ac, get_tof(sig7, sig1)), safe_div(ad, get_tof(sig7, sig2)), safe_div(ae, get_tof(sig7, sig3)), safe_div(ad, get_tof(sig7, sig4)), safe_div(ac, get_tof(sig7, sig5)), safe_div(ab, get_tof(sig7, sig6)), 0, safe_div(ab, get_tof(sig7, sig8))), dtype=np.float32)
        case 8: return np.array((safe_div(ab, get_tof(sig8, sig1)), safe_div(ac, get_tof(sig8, sig2)), safe_div(ad, get_tof(sig8, sig3)), safe_div(ae, get_tof(sig8, sig4)), safe_div(ad, get_tof(sig8, sig5)), safe_div(ac, get_tof(sig8, sig6)), safe_div(ab, get_tof(sig8, sig7)), 0), dtype=np.float32)
        case _: raise ValueError

def onebyeight(sensarray,which,diameter):
    return onetap(sensarray,which=which,diameter=diameter)

# ====================================================================
# FUNGSI HELPER & LOGIKA UTAMA (TIDAK ADA PERUBAHAN)
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
        properties=pika.BasicProperties(content_type='application/json', delivery_mode=2)
    )
    print(f"  📨  Payload BERHASIL dipublikasikan ke antrian '{RABBITMQ_RESULT_QUEUE}'.")

def get_guid_from_data(data_list):
    for item in data_list:
        if 'guidteensy' in item:
            return item['guidteensy']
    return None

def callback(ch, method, properties, body):
    global results_aggregator
    print(f"\n[+] Pesan baru diterima dari '{RABBITMQ_INPUT_QUEUE}'")
    try:
        message = json.loads(body)
        filename = message.get("filename")
        ketuk_ke = message.get("ketuk")
        sensor_referensi = message.get("which", ketuk_ke)

        if not all([filename, ketuk_ke]):
            print("  ❌  Pesan tidak valid. Diabaikan.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        downloaded_list = download_json_from_ftp(filename)
        
        if downloaded_list and isinstance(downloaded_list, list):
            guid_survey = get_guid_from_data(downloaded_list)
            if not guid_survey:
                try:
                    guid_survey = filename.split('_')[1]
                except IndexError:
                    guid_survey = "unknown_guid"
                print(f"  ⚠️  'guidteensy' tidak ditemukan, menggunakan GUID dari nama file: {guid_survey}")

            if guid_survey not in results_aggregator:
                results_aggregator[guid_survey] = {'ketuk_results': {}, 'original_filenames': {}}
                print(f"  🆕  Memulai sesi baru untuk GUID Survey: {guid_survey}")

            print(f"  🔬 Memulai perhitungan untuk ketuk #{ketuk_ke} dengan sensor referensi #{sensor_referensi}...")
            
            velo_result = onebyeight(downloaded_list, sensor_referensi, 0.3)
            velo_result = np.nan_to_num(velo_result, nan=0.0, posinf=0.0, neginf=0.0)
            
            session = results_aggregator[guid_survey]
            session['ketuk_results'][ketuk_ke] = velo_result.tolist()
            session['original_filenames'][ketuk_ke] = filename
            
            print(f"  👍  Perhitungan untuk GUID {guid_survey} ketuk #{ketuk_ke} selesai. ({len(session['ketuk_results'])}/8 terkumpul)")

            if len(session['ketuk_results']) == 8:
                print(f"\n✨ Semua 8 hasil untuk GUID {guid_survey} telah terkumpul! Memproses...")
                
                sorted_results = [session['ketuk_results'][i] for i in range(1, 9)]
                sorted_filenames = [session['original_filenames'][i] for i in range(1, 9)]
                
                result_filename = f"{guid_survey}.json"
                upload_result_to_ftp(result_filename, sorted_results)
                
                rmq_payload = {"filename": result_filename, "guid_survey": guid_survey, "matrix": sorted_results, "fileRow": sorted_filenames}
                publish_result_to_rmq(ch, rmq_payload)

                del results_aggregator[guid_survey]
                print(f"  ✅  Sesi untuk GUID {guid_survey} selesai dan dihapus.")
        else:
            print(f"  ❌  Data dari {filename} tidak valid atau bukan list. Diabaikan.")

    except Exception as e:
        import traceback
        print(f"  ❌  Terjadi kesalahan tak terduga saat pemrosesan: {e}")
        traceback.print_exc()
    
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