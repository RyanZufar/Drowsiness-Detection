# Real-Time Drowsiness Detection (Custom CNN + MediaPipe)

Sistem Deteksi Kantuk (Drowsiness Detection) *real-time* menggunakan **Python, OpenCV, MediaPipe** dengan arsitektur **CNN** yang ringan. 

Sistem ini dioptimalkan menggunakan sistem **Multithreading** untuk memisahkan beban kerja antara tracking kamera dan prediksi model, sehingga tidak menganggu proses tracking yang bisa membuat *lag* yang disebabkan oleh beban kerja secara bersamaan antara tracking dengan predikisi.

## 🗂️ Dataset
Kami menggunakan dataset [Drowsiness Detection](https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection) yang ada di Kaggle, yang di dalamnya terdapat 24.000 data dengan mata tertutup dan 24.000 data dengan mata terbuka.

## Fitur Utama
- **⚡ Multithreading Architecture:** Kamera berjalan di *Main Thread*, sementara model berjalan di *Background Thread*.
- **🧠 Model CNN:** Model dilatih menggunakan arsitektur CNN. Kami menghindari menggunakan model pretrained yang di train ulang agar komputasinya tidak terlalu berat. Untuk performa model kami memilik **Akurasi 0.9837** serta **Recall 0.99* pada data uji.
- **👀 Precision Eye Tracking:** Menggunakan *MediaPipe Face Mesh* untuk melacak titik koordinat mata secara dinamis dan akurat, bahkan dalam keadaan kepala bergerak.
- **⏱️ Logika Microsleep:** Dilengkapi sistem *Frame Buffer* untuk mencegah *False Alarm* saat pengguna hanya berkedip normal. Alarm hanya berbunyi jika mata tertutup selama beberapa *frame* berturut-turut (indikasi *microsleep*).

## requirements Sistem
- **Python:** Versi 3.10 atau 3.11 (Direkomendasikan agar kompatibel dengan TensorFlow 2.15).
- Webcam yang berfungsi dengan baik.

## Cara Instalasi

1. **Clone Repository ini**

2. **Buat Virtual Environment (Direkomendasikan)**
   
    python -m venv env
    env\Scripts\activate

3. **Install Dependencies**
    Install semua library yang dibutuhkan melalui requirements.txt:
    
    pip install -r requirements.txt

## Cara Penggunaan

1. Pastikan file model `drowsiness_CNNmodel.h5` dan file audio alarm `FAHHH (Meme Sound Effect).mp3` berada di folder yang sama dengan *script* utama.
2. Jalankan program dengan perintah:
    
    Script.py

3. Kamera akan terbuka. Sistem akan langsung mendeteksi wajah dan dalam terminal akan muncul akurasi dari model untuk hasil derteksi kedua mata dalam setiap frame yg diambil.
4. **Tekan tombol `q`** pada *keyboard* untuk keluar dan mematikan program.

## Kustomisasi
Kamu dapat menyesuaikan sensitivitas sistem dengan mengubah variabel berikut di dalam `Script.py`:

- `BATAS_TIDUR = 15`: Mengatur *Temporal Filtering*. Angka 15 berarti alarm akan berbunyi jika mata terdeteksi tertutup selama 15 *frame* berturut-turut. Naikkan angka ini jika sistem dirasa terlalu cepat berbunyi.
- `padding_x` dan `padding_y` (di dalam fungsi potong_mata): Saat ini disetel ke `5` pixel untuk menyesuaikan dengan dataset *training* asli (fokus pada bola mata).

## Note
Anda juga bisa melihat notebook yang kami gunakan dalam melatih model ini jika kedepannya ingin melakukan tuning terhadap model nya jika dianggap kurang memuaskan terhadap model nya
