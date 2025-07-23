# Layanan Pemrosesan Data Sensor Tomografi

Skrip ini berfungsi sebagai layanan latar belakang (daemon) yang secara otomatis memproses data sensor tomografi.

## Update

Perbaikan AI-written code pada function `gcc` dan `onetap`.

## Alur Kerja

1.  **Memonitor**: Terus-menerus memonitor folder `data_mentah` untuk file JSON baru.
2.  **Mengelompokkan**: Mencari satu set data yang terdiri dari 8 file JSON (`_1.json` hingga `_8.json`) yang berasal dari perangkat yang sama dan dibuat dalam rentang waktu 15 menit.
3.  **Memproses**: Jika satu set data yang valid ditemukan, skrip akan menjalankan algoritma _Generalized Cross-Correlation_ (GCC-PHAT) untuk menghitung matriks kecepatan 8x8.
4.  **Menyimpan**: Menyimpan file hasil pemrosesan (misalnya, `result_guid_timestamp.json`) di folder lokal `hasil_proses_lokal`.
5.  **Mengunggah**: Mengunggah file hasil yang sama ke server FTP.
6.  **Memberi Notifikasi**: Mengirimkan nama file hasil ke antrian (queue) RabbitMQ sebagai notifikasi bahwa pemrosesan telah selesai.
7.  **Membersihkan**: Menghapus 8 file JSON sumber dari folder `data_mentah` setelah berhasil diproses.

## Struktur Folder yang Dibutuhkan

Pastikan Anda membuat struktur folder berikut di direktori yang sama dengan skrip:

```
.
├── index.py                # Skrip utama ini
├── .env                    # File konfigurasi (dibuat manual)
├── requirements.txt        # File dependensi
├── data_mentah/            # Tempat menaruh file JSON sumber
├── hasil_proses_lokal/     # Hasil pemrosesan akan disimpan di sini
└── temp_data/              # Digunakan sementara oleh skrip
```

## Instalasi dan Pengaturan

1.  **Kloning Repositori (Opsional)**
    Jika kode ini berada dalam repositori git, kloning terlebih dahulu.

2.  **Buat Virtual Environment (Sangat Direkomendasikan)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows: ./venv/Scripts/activate
    ```

3.  **Instal Dependensi**
    Jalankan perintah berikut untuk menginstal semua pustaka Python yang diperlukan.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Buat File Konfigurasi `.env`**
    Buat file baru bernama `.env` di direktori yang sama dengan `main.py`. Salin konten dari file `.env.example` ke dalamnya dan isi dengan kredensial Anda yang sebenarnya.

    ```dotenv
    # Konten untuk file .env
    FTP_HOST=""
    FTP_PORT=2121
    FTP_USER=""
    FTP_PASSWORD=""
    FTP_FOLDER_HASIL="/result"

    RABBITMQ_HOST="r"
    RABBITMQ_PORT=5672
    RABBITMQ_USERNAME=""
    RABBITMQ_PASSWORD=""
    RABBITMQ_VHOST="/terawang"
    RABBITMQ_QUEUE="result_queue"
    ```

## Menjalankan Layanan

Untuk memulai layanan, cukup jalankan skrip utama dari terminal:

```bash
python index.py
```

Skrip akan mulai berjalan dan mencetak log aktivitasnya ke konsol. Untuk menghentikannya, tekan `Ctrl+C`.
