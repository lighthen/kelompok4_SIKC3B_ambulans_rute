# kelompok4_SIKC3B_ambulans_rute
Untuk kebutuhan mata kuliah kecerdasan buatan

# AmbuRoute — Sistem Pencarian Rute Tercepat Ambulans

Proyek UTS Kecerdasan Buatan
Kelompok 4 - SIKC 3B
1. Hendra Nurfatkhurrohman - 2407067
2. Widia Rina Lestari - 2407061
3. Nurizkya Mutiara Nisa - 2407031
4. Andea Asmanah - 2407062

AmbuRoute mencari rute tercepat ambulans dari lokasi kejadian ke rumah sakit
dengan mempertimbangkan kemacetan probabilistik dan pemblokiran jalan menggunakan
tiga metode AI yang terintegrasi secara berantai.

---

## Metode AI yang Digunakan

Ketiga metode bekerja secara berantai hasil satu metode menjadi input metode berikutnya:

### 1. First Order Logic (FOL)
Menentukan status tiap ruas: **Terblokir**, **Terbatas**, atau **Dapat Dilalui**.

Aturan R1–R7 diimplementasikan sebagai forward chaining manual. Contoh:
- R2: `Banjir(r) → Terblokir(r)`
- R3: `Kecelakaan(r) ∧ ¬Tertutup(r) ∧ ¬LajurLebih(r) → Terblokir(r)`
- R4: `Kecelakaan(r) ∧ ¬Tertutup(r) ∧ LajurLebih(r) → Terbatas(r)`

Menggunakan **Asumsi Dunia Tertutup**: ruas yang tidak terbukti Terblokir dianggap Dapat Dilalui.

### 2. Bayesian Network
Menghitung probabilitas macet `P(Macet=Ya)` untuk tiap ruas yang masih terbuka.

Struktur jaringan: `Jam`, `Cuaca`, `HariPasar`, `TipeRuas` → `Macet`

Menggunakan library `pgmpy` dengan fallback otomatis ke tabel CPT (dict Python)
jika pgmpy tidak tersedia. Interface fungsi tetap sama di kedua mode.

### 3. A\* Search
Mencari rute dengan waktu tempuh harapan minimum.

Heuristik: `h(n) = jarak_lurus(n, RS) / kecepatan_maks × 60` (menit) — admissible.

Implementasi sendiri dengan `heapq` (bukan `networkx.astar_path`).
Fungsi pembanding: BFS, Dijkstra, rute terpendek jarak.

---

## Prasyarat

- **Python 3.10 – 3.12** (disarankan). Python 3.13 belum tentu menyediakan wheel
  untuk versi dependensi yang di-pin (mis. `numpy==1.26.4`, `scipy==1.13.1`).
- Git.
- Tidak memerlukan internet saat aplikasi dijalankan (demo sepenuhnya offline).

---

## Cara Clone dan Menjalankan

### 1. Clone repositori

```bash
git clone https://github.com/lighthen/kelompok4_SIKC3B_ambulans_rute.git
cd kelompok4_SIKC3b_ambulans_rute
```

### 2. Buat dan aktifkan virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Jika `pgmpy` gagal dipasang (butuh `torch` ~124MB), sistem otomatis
> menggunakan fallback CPT dari `data/cpt_bawaan.json`. Semua fitur tetap berjalan.

### 4. Jalankan antarmuka Streamlit

```bash
streamlit run aplikasi.py
```

Buka browser di `http://localhost:8501`.

---

## Struktur Repositori

```
kelompok4_SIKC3b_ambulans_rute/
├── README.md
├── .gitignore
├── requirements.txt
├── pytest.ini
├── aplikasi.py                    # antarmuka Streamlit
├── data/
│   ├── graf_contoh.json           # graf 5 simpul (A, B, C, D, RS)
│   └── cpt_bawaan.json            # CPT Bayesian Network (ASUMSI)
├── sumber/
│   ├── basis_pengetahuan.py       # pusat pengetahuan: tetap, dinamis, turunan
│   ├── graf.py                    # memuat graf ke networkx
│   ├── fol.py                     # aturan R1–R7, forward chaining, jejak inferensi
│   ├── bayes.py                   # Bayesian Network + fallback
│   ├── biaya.py                   # biaya ruas = waktu harapan
│   ├── astar.py                   # A*, BFS, Dijkstra, rute terpendek jarak
│   ├── pipeline.py                # integrasi FOL → Bayes → A*
│   └── latih_cpt.py               # melatih CPT dari CSV (opsional, Fase 4)
└── tes/
    ├── tes_validasi_manual.py     # Checkpoint 1
    ├── tes_fol.py
    ├── tes_bayes.py
    ├── tes_astar.py
    ├── tes_basis_pengetahuan.py
    ├── tes_graf_biaya.py
    ├── tes_integrasi.py           # Checkpoint 2 (S0–S5)
    └── tes_smoke_aplikasi.py      # uji asap antarmuka Streamlit
```


## Penjelasan Basis Pengetahuan

Seluruh pengetahuan sistem dikumpulkan dalam kelas `BasisPengetahuan`
(`sumber/basis_pengetahuan.py`) yang memisahkan tiga jenis pengetahuan.
Modul lain hanya membaca dari sini, tidak menyimpan data sendiri.

| Jenis | Isi | Sifat |
|---|---|---|
| **Tetap** | Graf jalan, CPT Bayesian, aturan FOL R1–R7 | Dimuat sekali dari file, tidak berubah |
| **Dinamis** | Fakta lapangan (kecelakaan/banjir), konteks (jam/cuaca/pasar), titik awal/tujuan | Berubah mengikuti input pengguna |
| **Turunan** | Status ruas dari FOL, P(macet), biaya, jejak inferensi, rute A* | Dihitung oleh pipeline, **dihapus otomatis** saat fakta dinamis berubah |

**Aturan penting:** Mengubah fakta dinamis selalu menghapus pengetahuan turunan,
sehingga hasil lama tidak pernah dipakai ulang pada pencarian berikutnya.

---

## Alur Data Sistem

```
Input: titik awal, RS tujuan, jam, cuaca, hari pasar, laporan kejadian
        ↓
FOL (forward chaining R1–R7)
→ status ruas: Terblokir / Terbatas / Dapat Dilalui
→ jejak inferensi
        ↓
Graf diperbarui: ruas Terblokir dihapus, ruas Terbatas × waktu 2
        ↓
Bayesian Network → P(macet) per ruas
        ↓
Biaya ruas = t_lancar × (1 + P(macet) × 2)
        ↓
A* Search → rute tercepat
        ↓
Output: rute, waktu harapan, jejak FOL, perbandingan BFS/Dijkstra/jarak
```

---

## ASUMSI

Semua asumsi diberi label `# ASUMSI` di kode dan di `data/cpt_bawaan.json`.

| Asumsi | Nilai | Lokasi |
|---|---|---|
| CPT Bayesian Network | Seluruh tabel 16 entri | `data/cpt_bawaan.json` |
| CPT tipe lokal | `P_lokal = max(0.02, min(0.98, P_utama - 0.15))` | `data/cpt_bawaan.json` |
| Kecepatan lancar semua ruas | 40 km/jam | `data/graf_contoh.json` |
| Faktor pengali kemacetan | 2 (dalam rumus biaya) | `sumber/biaya.py` |
| Faktor pengali ruas Terbatas | 2 (dari FOL R4) | `sumber/biaya.py` |
| Koordinat simpul | Dipilih agar heuristik A* admissible | `data/graf_contoh.json` |
| Graf jalan | 5 simpul contoh, bukan peta nyata Indramayu | `data/graf_contoh.json` |

---

## Keterbatasan

- Graf saat ini hanya 5 simpul contoh (A, B, C, D, RS), bukan peta nyata.
- CPT Bayesian Network menggunakan asumsi, bukan data historis lapangan.
- FOL hanya mendukung klausa Horn dengan variabel yang diinstansiasi ke konstanta.
- Peta nyata (OSMnx + Folium) belum diimplementasikan (Fase 4, opsional).
- `latih_cpt.py` (pelatihan dari CSV) belum diimplementasikan (Fase 4, opsional).

---

## Hasil Uji Singkat

| Skenario | Rute A* | Waktu (mnt) | Catatan |
|---|---|---|---|
| S0 Validasi manual | A→C→D→RS | **12.75** | Cocok tabel perhitungan manual |
| S1 Normal+Cerah+Tidak | A→B→RS | 9.00 | Kondisi normal |
| S2 Sibuk+Hujan+Ya | A→B→RS | 21.75 | Lebih lambat dari S1 |
| S3 Kecelakaan B-RS | A→C→D→RS | 25.35 | Rute dialihkan |
| S4 Kecelakaan A-C | A→B→RS | 9.00 | A-C terbatas (×2) |
| S5 Banjir B-RS+D-RS | — | — | RS terisolasi, pesan jelas |

---
