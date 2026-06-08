# ⚙️ SampahClassifier - Workflow CI/CD & MLProject

Repositori ini merupakan **Bagian 2** dari proyek *Machine Learning Operations* (MLOps) untuk klasifikasi sampah daur ulang. Repositori ini memuat konfigurasi penuh untuk otomasi siklus pelatihan model dan proses *deployment* instan menggunakan integrasi **MLflow** dan **GitHub Actions**.

---

## 📂 Struktur Repositori

```text
Workflow_CI/
├── .workflow/
│   └── main.yml                       # Skrip CI/CD GitHub Actions
└── MLProject/
    ├── MLProject                      # Konfigurasi entry-point MLflow
    ├── conda.yaml                     # Definisi env dependensi untuk eksekusi pipeline
    ├── modelling.py                   # Skrip training arsitektur MobileNetV2
    ├── sampah-daur-ulang_preprocessing/ 
    └── Tautan ke Docker Hub.txt       # Link rujukan Docker Image untuk inferensi
```

## 🚀 Fitur Automasi Canggih (CI/CD)
Setiap kali ada pembaruan kode yang di-*push* ke repositori ini, *server* GitHub Actions akan secara otonom membangkitkan mesin virtual untuk melakukan tahapan berikut:

1. **Inisiasi MLflow Run:** Mengeksekusi file `MLProject` dan membaca definisi `modelling.py`.
2. **Training Terotomatisasi:** Memulai pengunduhan dataset terbaru dari Kaggle, lalu melatih model klasifikasi sampah selama beberapa epochs secara *headless*.
3. **Pencatatan Model:** Semua parameter, grafik loss, dan akurasi dicatat (*tracking*) ke dalam MLflow Registry.
4. **Containerization:** Memanfaatkan `mlflow models build-docker` untuk langsung membungkus model biner `.keras` ke dalam sebuah kontainer web (*Flask*).
5. **Zero-Touch Deployment:** Menandai (*tagging*) dan mendorong (*push*) *image* tersebut langsung ke server **Docker Hub** publik.

Sistem *pipeline* ini memastikan model produksi yang digunakan oleh klien selalu merupakan versi mutakhir yang divalidasi.
