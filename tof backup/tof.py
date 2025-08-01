import numpy as np
import scipy.signal
import scipy.ndimage
import json
import codecs
import pika
import ftplib
import io
import time

# --- KONFIGURASI (Sama seperti skrip pengunggah) ---
# RabbitMQ
RABBITMQ_HOST = "rmq230.pptik.id"
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = "terawang"
RABBITMQ_PASSWORD = "Terawang@#2025"
RABBITMQ_VHOST = "/terawang"
RABBITMQ_INPUT_QUEUE = "terawangDataRow"   # Antrian untuk menerima data
RABBITMQ_RESULT_QUEUE = "terawangHasil"    # Antrian baru untuk mengirim hasil

# FTP
FTP_HOST = "ftp-sth.pptik.id"
FTP_USER = "terawang"
FTP_PASSWORD = "Terawang@#2025"
FTP_PORT = 2121
FTP_SOURCE_FOLDER = "/terawang"

# Variabel global untuk mengumpulkan hasil
results_aggregator = {}

# ====================================================================
# SEMUA FUNGSI PERHITUNGAN DI BAWAH INI TIDAK DIUBAH SAMA SEKALI
# ====================================================================

def gcc(sig, refsig, fs=1000000, interp=128, max_tau=None, CCType="PHAT", timestamp=None):
    n = len(sig)
    sig -= np.mean(sig, axis=0)
    refsig -= np.mean(refsig, axis=0)
    SIG = np.fft.rfft(sig, axis=0, n=n)
    REFSIG = np.fft.rfft(refsig, axis=0, n=n)
    CONJ = np.conj(REFSIG)
    R = np.multiply(SIG, CONJ)
    match CCType:
        case "CC" | "cc":
            WEIGHT = 1
        case "PHAT" | "Phat" | "phat":
            CCType = "PHAT"
            WEIGHT = 1/np.abs(R)
        case "SCOT" | "Scot" | "scot":
            CCType = "SCOT"
            WEIGHT = 1/np.sqrt(SIG*np.conj(SIG)*REFSIG*CONJ)
        case "ROTH" | "Roth" | "roth":
            CCType = "ROTH"
            WEIGHT = 1/(SIG*np.conj(SIG))
        case _:
            CCType = "CC"
            WEIGHT = 1
    Integ = np.multiply(R, WEIGHT)
    cc = np.fft.irfft(a=Integ, axis=0, n=n)
    lags = scipy.signal.correlation_lags(len(sig), len(refsig), mode='same')
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    smallcc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    smallcc /= np.max(cc)
    shift = np.argmax(smallcc) - max_shift
    cc = scipy.ndimage.shift(cc, len(cc)/2, mode="grid-wrap", order=5)
    cc /= np.max(cc)
    tau = shift / float(interp * fs)
    if timestamp is not None:
        peaktimestamp = timestamp[np.argmax(cc)]
        timestamp = scipy.ndimage.shift(timestamp, len(timestamp)/2, mode="grid-wrap", order=5)
        a = timestamp[0]
        b = timestamp[max_shift]
        c = timestamp[-max_shift-1]
        d = timestamp[-1]
        print(peaktimestamp)
        if a > peaktimestamp >= b:
            tau = int(peaktimestamp - a)
        else:
            tau = int(-peaktimestamp + c)
        tau /= 1000000
    tau /= 10
    return np.abs(tau), cc, lags

def onetap(sigdict, which, diameter):
    sig1, sig2, sig3, sig4 = sigdict["value1"], sigdict["value2"], sigdict["value3"], sigdict["value4"]
    sig5, sig6, sig7, sig8 = sigdict["value5"], sigdict["value6"], sigdict["value7"], sigdict["value8"]
    timestamp = sigdict["timestamp"]
    radius = diameter/2
    ab = radius * 0.76536686473
    ac = radius * 1.41421356237
    ad = radius * 1.84775906502
    ae = float(diameter)
    match which:
        case 1:
            tof12, tof13, tof14, tof15 = gcc(sig1, sig2)[0], gcc(sig1, sig3)[0], gcc(sig1, sig4)[0], gcc(sig1, sig5)[0]
            tof16, tof17, tof18 = gcc(sig1, sig6)[0], gcc(sig1, sig7)[0], gcc(sig1, sig8)[0]
            velo12, velo13, velo14, velo15 = ab/tof12, ac/tof13, ad/tof14, ae/tof15
            velo16, velo17, velo18 = ad/tof16, ac/tof17, ab/tof18
            return np.array((0, velo12, velo13, velo14, velo15, velo16, velo17, velo18), dtype=np.float32)
        case 2:
            tof21, tof23, tof24, tof25 = gcc(sig2, sig1)[0], gcc(sig2, sig3)[0], gcc(sig2, sig4)[0], gcc(sig2, sig5)[0]
            tof26, tof27, tof28 = gcc(sig2, sig6)[0], gcc(sig2, sig7)[0], gcc(sig2, sig8)[0]
            velo21, velo23, velo24, velo25 = ab/tof21, ab/tof23, ac/tof24, ad/tof25
            velo26, velo27, velo28 = ae/tof26, ad/tof27, ac/tof28
            return np.array((velo21, 0, velo23, velo24, velo25, velo26, velo27, velo28), dtype=np.float32)
        case 3:
            tof31, tof32, tof34, tof35 = gcc(sig3,sig1)[0], gcc(sig3,sig2)[0], gcc(sig3,sig4)[0], gcc(sig3,sig5)[0]
            tof36, tof37, tof38 = gcc(sig3,sig6)[0], gcc(sig3,sig7)[0], gcc(sig3,sig8)[0]
            velo31, velo32, velo34, velo35 = ac/tof31, ab/tof32, ab/tof34, ac/tof35
            velo36, velo37, velo38 = ad/tof36, ae/tof37, ad/tof38
            return np.array((velo31, velo32, 0, velo34, velo35, velo36, velo37, velo38), dtype=np.float32)
        case 4:
            tof41, tof42, tof43, tof45 = gcc(sig4,sig1)[0], gcc(sig4,sig2)[0], gcc(sig4,sig3)[0], gcc(sig4,sig5)[0]
            tof46, tof47, tof48 = gcc(sig4,sig6)[0], gcc(sig4,sig7)[0], gcc(sig4,sig8)[0]
            velo41, velo42, velo43, velo45 = ad/tof41, ac/tof42, ab/tof43, ab/tof45
            velo46, velo47, velo48 = ac/tof46, ad/tof47, ae/tof48
            return np.array((velo41, velo42, velo43, 0, velo45, velo46, velo47, velo48), dtype=np.float32)
        case 5:
            tof51, tof52, tof53, tof54 = gcc(sig5,sig1)[0], gcc(sig5,sig2)[0], gcc(sig5,sig3)[0], gcc(sig5,sig4)[0]
            tof56, tof57, tof58 = gcc(sig5,sig6)[0], gcc(sig5,sig7)[0], gcc(sig5,sig8)[0]
            velo51, velo52, velo53, velo54 = ae/tof51, ad/tof52, ac/tof53, ab/tof54
            velo56, velo57, velo58 = ab/tof56, ac/tof57, ad/tof58
            return np.array((velo51, velo52, velo53, velo54, 0, velo56, velo57, velo58), dtype=np.float32)
        case 6:
            tof61, tof62, tof63, tof64 = gcc(sig6,sig1)[0], gcc(sig6,sig2)[0], gcc(sig6,sig3)[0], gcc(sig6,sig4)[0]
            tof65, tof67, tof68 = gcc(sig6,sig5)[0], gcc(sig6,sig7)[0], gcc(sig6,sig8)[0]
            velo61, velo62, velo63, velo64 = ad/tof61, ae/tof62, ad/tof63, ac/tof64
            velo65, velo67, velo68 = ab/tof65, ab/tof67, ac/tof68
            return np.array((velo61, velo62, velo63, velo64, velo65, 0, velo67, velo68), dtype=np.float32)
        case 7:
            tof71, tof72, tof73, tof74 = gcc(sig7,sig1)[0], gcc(sig7,sig2)[0], gcc(sig7,sig3)[0], gcc(sig7,sig4)[0]
            tof75, tof76, tof78 = gcc(sig7,sig5)[0], gcc(sig7,sig6)[0], gcc(sig7,sig8)[0]
            velo71, velo72, velo73, velo74 = ac/tof71, ad/tof72, ae/tof73, ad/tof74
            velo75, velo76, velo78 = ac/tof75, ab/tof76, ab/tof78
            return np.array((velo71, velo72, velo73, velo74, velo75, velo76, 0, velo78), dtype=np.float32)
        case 8:
            tof81, tof82, tof83, tof84 = gcc(sig8,sig1)[0], gcc(sig8,sig2)[0], gcc(sig8,sig3)[0], gcc(sig8,sig4)[0]
            tof85, tof86, tof87 = gcc(sig8,sig5)[0], gcc(sig8,sig6)[0], gcc(sig8,sig7)[0]
            velo81, velo82, velo83, velo84 = ab/tof81, ac/tof82, ad/tof83, ae/tof84
            velo85, velo86, velo87 = ad/tof85, ac/tof86, ab/tof87
            return np.array((velo81, velo82, velo83, velo84, velo85, velo86, velo87, 0), dtype=np.float32)
        case _:
            raise ValueError("Invalid number. Expected between 1 and 8")

def onebyeight(sensarray, which, diameter):
    return onetap(sensarray, which=which, diameter=diameter)

# ====================================================================
# BAGIAN BARU & MODIFIKASI
# ====================================================================

def download_json_from_ftp(filename):
    """Mengunduh file JSON dari FTP dan membacanya."""
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
            print(f"  ✅  File '{filename}' berhasil diunduh dan dibaca.")
            return data
    except ftplib.all_errors as e:
        print(f"  ❌  GAGAL mengunduh dari FTP: {e}")
        return None

def publish_result(channel, result_data):
    """Mempublikasikan hasil akhir ke antrian RabbitMQ."""
    payload = json.dumps(result_data, indent=4)
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_RESULT_QUEUE,
        body=payload,
        properties=pika.BasicProperties(
            content_type='application/json',
            delivery_mode=2,
        )
    )
    print("\n🚀 Matriks 8x8 akhir BERHASIL dipublikasikan ke antrian 'terawangHasil'.")

def callback(ch, method, properties, body):
    """Fungsi yang dieksekusi setiap kali ada pesan masuk."""
    global results_aggregator
    
    print(f"\n[+] Pesan baru diterima dari '{RABBITMQ_INPUT_QUEUE}'")
    try:
        message = json.loads(body)
        filename = message.get("filename")
        ketuk_ke = message.get("ketuk")

        if not filename or not ketuk_ke:
            print("  ❌  Pesan tidak valid (kurang filename atau ketuk). Pesan diabaikan.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        # 1. Unduh file dari FTP
        downloaded_list = download_json_from_ftp(filename)
        
        if downloaded_list and isinstance(downloaded_list, list):
            # =================================================================
            # === PERBAIKAN: Mengubah format data dari List ke Dictionary ===
            # =================================================================
            print("  🔄  Mengubah format data dari List menjadi Dictionary...")
            data_dict = {}
            for item in downloaded_list:
                data_dict.update(item)
            
            # 2. Lakukan perhitungan (logika inti yang tidak diubah)
            print(f"  🔬 Memulai perhitungan untuk ketuk #{ketuk_ke}...")
            velo_result = onebyeight(data_dict, ketuk_ke, 0.3)
            
            # 3. Simpan hasil sementara
            results_aggregator[ketuk_ke] = velo_result
            print(f"  👍  Perhitungan untuk ketuk #{ketuk_ke} selesai. ({len(results_aggregator)}/8 terkumpul)")

            # 4. Periksa apakah sudah 8 hasil terkumpul
            if len(results_aggregator) == 8:
                print("\n✨ Semua 8 hasil telah terkumpul! Memproses matriks akhir...")
                sorted_results = [results_aggregator[i] for i in range(1, 9)]
                veloall = np.vstack(sorted_results)
                beloall = veloall.tolist()
                file_path = "hasil_akhir.json"
                with codecs.open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(beloall, f, separators=(',', ':'), sort_keys=True, indent=4)
                print(f"  💾  Matriks akhir disimpan di '{file_path}'.")
                publish_result(ch, beloall)
                results_aggregator.clear()
                print("  🔄  Agregator direset, siap untuk 8 ketukan berikutnya.")
        else:
            print(f"  ❌  Data dari {filename} tidak valid atau bukan list. Melompati proses.")

    except json.JSONDecodeError:
        print("  ❌  Gagal mem-parsing pesan JSON.")
    except Exception as e:
        print(f"  ❌  Terjadi kesalahan tak terduga saat pemrosesan: {e}")
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    """Fungsi utama untuk menjalankan koneksi dan consumer RabbitMQ."""
    credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        virtual_host=RABBITMQ_VHOST,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300
    )
    
    print("Menghubungkan ke RabbitMQ...")
    while True:
        try:
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            channel.queue_declare(queue=RABBITMQ_INPUT_QUEUE, durable=True)
            channel.queue_declare(queue=RABBITMQ_RESULT_QUEUE, durable=True)
            
            channel.basic_qos(prefetch_count=1)
            
            channel.basic_consume(
                queue=RABBITMQ_INPUT_QUEUE,
                on_message_callback=callback
            )

            print(f"✅ Terhubung! Menunggu pesan di antrian '{RABBITMQ_INPUT_QUEUE}'...")
            print("Tekan CTRL+C untuk berhenti.")
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