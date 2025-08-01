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


# Variabel global untuk mengumpulkan hasil dari sesi yang berbeda
# Key: guid_survey, Value: { 'ketuk_results': {}, 'original_filenames': {} }
results_aggregator = {}

# ====================================================================
# FUNGSI PERHITUNGAN INTI
# ====================================================================

def gcc(sig, refsig, fs=1000000, interp=128, max_tau=None, CCType="PHAT", timestamp=None):
    n = len(sig)
    # Pastikan input adalah numpy array float
    sig = np.array(sig, dtype=np.float64)
    refsig = np.array(refsig, dtype=np.float64)
    
    sig -= np.mean(sig, axis=0)
    refsig -= np.mean(refsig, axis=0)
    
    SIG = np.fft.rfft(sig, axis=0, n=n)
    REFSIG = np.fft.rfft(refsig, axis=0, n=n)
    
    CONJ = np.conj(REFSIG)
    
    R = np.multiply(SIG,CONJ)
    
    # Mencegah pembagian dengan nol
    R_abs = np.abs(R)
    R_abs[R_abs == 0] = 1e-10

    match CCType:
        case "CC" | "cc":
            WEIGHT = 1
        case "PHAT" | "Phat" | "phat":
            CCType = "PHAT"
            WEIGHT = 1/R_abs
        case "SCOT" | "Scot" | "scot":
            CCType = "SCOT"
            # Mencegah sqrt dari nol
            denom = np.sqrt(np.abs(SIG*np.conj(SIG)*REFSIG*CONJ))
            denom[denom == 0] = 1e-10
            WEIGHT = 1/denom
        case "ROTH" | "Roth" | "roth":
            CCType = "ROTH"
            denom = np.abs(SIG*np.conj(SIG))
            denom[denom == 0] = 1e-10
            WEIGHT = 1/denom
        case _:
            CCType = "CC"
            WEIGHT = 1
    
    Integ = np.multiply(R,WEIGHT)
    
    cc = np.fft.irfft(a=Integ, axis=0, n=n)
    
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(int(interp * fs * max_tau), max_shift)
        
    
    smallcc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    # Handle kasus di mana cc adalah semua nol
    if np.max(np.abs(cc)) > 0:
        smallcc /= np.max(np.abs(cc))

    shift = np.argmax(smallcc) - max_shift
    
    tau = shift / float(interp * fs)
    
    if timestamp is not None:
        # Logika timestamp ini dipertahankan sesuai aslinya, meskipun tidak digunakan saat ini
        cc_shifted = scipy.ndimage.shift(cc, len(cc)/2, mode="grid-wrap", order = 5)
        peaktimestamp = timestamp[np.argmax(cc_shifted)]
        timestamp_shifted = scipy.ndimage.shift(timestamp, len(timestamp)/2, mode="grid-wrap", order = 5)
        a = timestamp_shifted[0]
        b = timestamp_shifted[max_shift]
        c = timestamp_shifted[-max_shift-1]
        if a > peaktimestamp >=  b:
            tau = int(peaktimestamp - a) # in micros
        else:
            tau = int(-peaktimestamp + c) # in micros, negative
        tau /= 1000000 # convert to seconds

    tau /= 10
    
    return np.abs(tau), cc, None

def onetap(sigdict, which, diameter):
    sig1, sig2, sig3, sig4 = sigdict["value1"], sigdict["value2"], sigdict["value3"], sigdict["value4"]
    sig5, sig6, sig7, sig8 = sigdict["value5"], sigdict["value6"], sigdict["value7"], sigdict["value8"]
    
    radius = diameter/2
    ab = radius * 0.76536686473
    ac = radius * 1.41421356237
    ad = radius * 1.84775906502
    ae = float(diameter)
    
    def calc_velocity(dist, tof):
        if tof > 0:
            return dist / tof
        return 0

    match which:
        case 1:
            tof12, tof13, tof14, tof15 = gcc(sig1, sig2)[0], gcc(sig1, sig3)[0], gcc(sig1, sig4)[0], gcc(sig1, sig5)[0]
            tof16, tof17, tof18 = gcc(sig1, sig6)[0], gcc(sig1, sig7)[0], gcc(sig1, sig8)[0]
            velo12, velo13, velo14, velo15 = calc_velocity(ab, tof12), calc_velocity(ac, tof13), calc_velocity(ad, tof14), calc_velocity(ae, tof15)
            velo16, velo17, velo18 = calc_velocity(ad, tof16), calc_velocity(ac, tof17), calc_velocity(ab, tof18)
            return np.array((0, velo12, velo13, velo14, velo15, velo16, velo17, velo18), dtype=np.float32)
        case 2:
            tof21, tof23, tof24, tof25 = gcc(sig2, sig1)[0], gcc(sig2, sig3)[0], gcc(sig2, sig4)[0], gcc(sig2, sig5)[0]
            tof26, tof27, tof28 = gcc(sig2, sig6)[0], gcc(sig2, sig7)[0], gcc(sig2, sig8)[0]
            velo21, velo23, velo24, velo25 = calc_velocity(ab, tof21), calc_velocity(ab, tof23), calc_velocity(ac, tof24), calc_velocity(ad, tof25)
            velo26, velo27, velo28 = calc_velocity(ae, tof26), calc_velocity(ad, tof27), calc_velocity(ac, tof28)
            return np.array((velo21, 0, velo23, velo24, velo25, velo26, velo27, velo28), dtype=np.float32)
        case 3:
            tof31, tof32, tof34, tof35 = gcc(sig3,sig1)[0], gcc(sig3,sig2)[0], gcc(sig3,sig4)[0], gcc(sig3,sig5)[0]
            tof36, tof37, tof38 = gcc(sig3,sig6)[0], gcc(sig3,sig7)[0], gcc(sig3,sig8)[0]
            velo31, velo32, velo34, velo35 = calc_velocity(ac, tof31), calc_velocity(ab, tof32), calc_velocity(ab, tof34), calc_velocity(ac, tof35)
            velo36, velo37, velo38 = calc_velocity(ad, tof36), calc_velocity(ae, tof37), calc_velocity(ad, tof38)
            return np.array((velo31, velo32, 0, velo34, velo35, velo36, velo37, velo38), dtype=np.float32)
        case 4:
            tof41, tof42, tof43, tof45 = gcc(sig4,sig1)[0], gcc(sig4,sig2)[0], gcc(sig4,sig3)[0], gcc(sig4,sig5)[0]
            tof46, tof47, tof48 = gcc(sig4,sig6)[0], gcc(sig4,sig7)[0], gcc(sig4,sig8)[0]
            velo41, velo42, velo43, velo45 = calc_velocity(ad, tof41), calc_velocity(ac, tof42), calc_velocity(ab, tof43), calc_velocity(ab, tof45)
            velo46, velo47, velo48 = calc_velocity(ac, tof46), calc_velocity(ad, tof47), calc_velocity(ae, tof48)
            return np.array((velo41, velo42, velo43, 0, velo45, velo46, velo47, velo48), dtype=np.float32)
        case 5:
            tof51, tof52, tof53, tof54 = gcc(sig5,sig1)[0], gcc(sig5,sig2)[0], gcc(sig5,sig3)[0], gcc(sig5,sig4)[0]
            tof56, tof57, tof58 = gcc(sig5,sig6)[0], gcc(sig5,sig7)[0], gcc(sig5,sig8)[0]
            velo51, velo52, velo53, velo54 = calc_velocity(ae, tof51), calc_velocity(ad, tof52), calc_velocity(ac, tof53), calc_velocity(ab, tof54)
            velo56, velo57, velo58 = calc_velocity(ab, tof56), calc_velocity(ac, tof57), calc_velocity(ad, tof58)
            return np.array((velo51, velo52, velo53, velo54, 0, velo56, velo57, velo58), dtype=np.float32)
        case 6:
            tof61, tof62, tof63, tof64 = gcc(sig6,sig1)[0], gcc(sig6,sig2)[0], gcc(sig6,sig3)[0], gcc(sig6,sig4)[0]
            tof65, tof67, tof68 = gcc(sig6,sig5)[0], gcc(sig6,sig7)[0], gcc(sig6,sig8)[0]
            velo61, velo62, velo63, velo64 = calc_velocity(ad, tof61), calc_velocity(ae, tof62), calc_velocity(ad, tof63), calc_velocity(ac, tof64)
            velo65, velo67, velo68 = calc_velocity(ab, tof65), calc_velocity(ab, tof67), calc_velocity(ac, tof68)
            return np.array((velo61, velo62, velo63, velo64, velo65, 0, velo67, velo68), dtype=np.float32)
        case 7:
            tof71, tof72, tof73, tof74 = gcc(sig7,sig1)[0], gcc(sig7,sig2)[0], gcc(sig7,sig3)[0], gcc(sig7,sig4)[0]
            tof75, tof76, tof78 = gcc(sig7,sig5)[0], gcc(sig7,sig6)[0], gcc(sig7,sig8)[0]
            velo71, velo72, velo73, velo74 = calc_velocity(ac, tof71), calc_velocity(ad, tof72), calc_velocity(ae, tof73), calc_velocity(ad, tof74)
            velo75, velo76, velo78 = calc_velocity(ac, tof75), calc_velocity(ab, tof76), calc_velocity(ab, tof78)
            return np.array((velo71, velo72, velo73, velo74, velo75, velo76, 0, velo78), dtype=np.float32)
        case 8:
            tof81, tof82, tof83, tof84 = gcc(sig8,sig1)[0], gcc(sig8,sig2)[0], gcc(sig8,sig3)[0], gcc(sig8,sig4)[0]
            tof85, tof86, tof87 = gcc(sig8,sig5)[0], gcc(sig8,sig6)[0], gcc(sig8,sig7)[0]
            velo81, velo82, velo83, velo84 = calc_velocity(ab, tof81), calc_velocity(ac, tof82), calc_velocity(ad, tof83), calc_velocity(ae, tof84)
            velo85, velo86, velo87 = calc_velocity(ad, tof85), calc_velocity(ac, tof86), calc_velocity(ab, tof87)
            return np.array((velo81, velo82, velo83, velo84, velo85, velo86, velo87, 0), dtype=np.float32)
        case _:
            raise ValueError("Invalid number. Expected between 1 and 8")

def onebyeight(sensarray, which, diameter):
    return onetap(sensarray, which=which, diameter=diameter)

# ====================================================================
# FUNGSI HELPER
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
    """Mengunggah file hasil JSON ke folder /result di FTP."""
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
    """Mempublikasikan payload hasil ke antrian RabbitMQ yang ditentukan."""
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
    """Mencari dan mengembalikan guidteensy dari list of dictionaries."""
    for item in data_list:
        if 'guidteensy' in item:
            return item['guidteensy']
    return None

# ====================================================================
# LOGIKA UTAMA
# ====================================================================

def callback(ch, method, properties, body):
    """Fungsi yang dieksekusi setiap kali ada pesan masuk."""
    global results_aggregator
    
    print(f"\n[+] Pesan baru diterima dari '{RABBITMQ_INPUT_QUEUE}'")
    try:
        message = json.loads(body)
        filename = message.get("filename")
        ketuk_ke = message.get("ketuk")

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

            data_dict = {k: v for d in downloaded_list for k, v in d.items()}
            
            print(f"  🔬 Memulai perhitungan untuk ketuk #{ketuk_ke}...")
            velo_result = onebyeight(data_dict, ketuk_ke, 0.3)
            velo_result = np.nan_to_num(velo_result, nan=0.0, posinf=0.0, neginf=0.0)
            
            # --- PERUBAHAN DI SINI ---
            # Mengonversi hasil float menjadi integer untuk menghapus desimal
            velo_result = velo_result.astype(np.int32)
            
            session = results_aggregator[guid_survey]
            session['ketuk_results'][ketuk_ke] = velo_result.tolist()
            # Menyimpan nama file asli
            session['original_filenames'][ketuk_ke] = filename
            
            print(f"  👍  Perhitungan untuk GUID {guid_survey} ketuk #{ketuk_ke} selesai. ({len(session['ketuk_results'])}/8 terkumpul)")

            if len(session['ketuk_results']) == 8:
                print(f"\n✨ Semua 8 hasil untuk GUID {guid_survey} telah terkumpul! Memproses...")
                
                sorted_results = [session['ketuk_results'][i] for i in range(1, 9)]
                # Mengambil dan mengurutkan nama file asli
                sorted_filenames = [session['original_filenames'][i] for i in range(1, 9)]
                
                result_filename = f"{guid_survey}.json"
                
                upload_result_to_ftp(result_filename, sorted_results)
                
                rmq_payload = {
                    "filename": result_filename,
                    "guid_survey": guid_survey,
                    "matrix": sorted_results,
                    "fileRow": sorted_filenames  # Menambahkan field baru ke payload
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
            print(f"Kesalahan tidak terduga: {e}. Restarting...")
            time.sleep(5)

if __name__ == '__main__':
    main()
