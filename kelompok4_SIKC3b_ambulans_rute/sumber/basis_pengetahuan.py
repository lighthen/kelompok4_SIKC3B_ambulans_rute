"""
basis_pengetahuan.py — Pusat pengetahuan sistem AmbuRoute.

Seluruh pengetahuan sistem dikumpulkan di sini agar mudah dijelaskan saat
presentasi dan mudah dipindahkan ke laporan. Modul lain (graf.py, fol.py,
bayes.py, pipeline.py) membaca dari kelas ini dan tidak menyimpan data sendiri.

Ada tiga jenis pengetahuan yang dipisahkan dengan jelas:

1. TETAP  : Data yang dimuat sekali dan tidak berubah selama sistem berjalan.
            Contoh: struktur graf jalan, CPT Bayesian, daftar aturan FOL.

2. DINAMIS: Data yang berubah mengikuti masukan pengguna saat aplikasi berjalan.
            Contoh: fakta lapangan (kecelakaan/banjir), konteks (jam/cuaca/pasar),
            titik awal dan RS tujuan.

3. TURUNAN: Hasil komputasi sistem dari pengetahuan tetap + dinamis.
            Contoh: status ruas dari FOL, P(macet) per ruas, biaya per ruas,
            jejak inferensi, rute A*.
            Dihapus otomatis saat fakta dinamis berubah agar tidak terpakai ulang.
"""

import json
import os
from datetime import datetime


class BasisPengetahuan:
    """
    Kelas utama yang menyimpan seluruh pengetahuan sistem AmbuRoute.

    Pengetahuan dibagi tiga:
    - tetap    : graf, CPT Bayesian, aturan FOL (dimuat dari file)
    - dinamis  : fakta lapangan, konteks, titik awal/tujuan (dari input pengguna)
    - turunan  : hasil inferensi FOL, P(macet), biaya, rute (dari komputasi pipeline)
    """

    def __init__(self):
        """Inisialisasi basis pengetahuan kosong."""
        # --- PENGETAHUAN TETAP ---
        # Data yang dimuat sekali dari file dan tidak berubah saat sistem berjalan.
        self.tetap = {
            "simpul": [],          # daftar simpul dengan id, nama, koordinat
            "ruas": [],            # daftar ruas dengan id, dari, ke, jarak, tipe, kecepatan
            "cpt": [],             # tabel CPT Bayesian (16 entri)
            "variabel_bayes": {},  # nama variabel dan nilai-nilainya
            "aturan_fol": self._daftar_aturan_fol(),  # R1-R7 beserta teks notasi
        }

        # --- PENGETAHUAN DINAMIS ---
        # Data yang berubah mengikuti masukan pengguna.
        # Mengubah bagian ini HARUS menghapus pengetahuan turunan.
        self.dinamis = {
            "fakta": set(),        # set of tuple, misal: ("Kecelakaan", "B-RS")
            "konteks": {           # kondisi lingkungan saat ini
                "jam": None,
                "cuaca": None,
                "hari_pasar": None,
            },
            "titik_awal": None,
            "tujuan": None,
            "p_macet_manual": {},  # override P(macet) per ruas (mode validasi manual)
        }

        # --- PENGETAHUAN TURUNAN ---
        # Hasil komputasi dari pipeline. Dihapus saat fakta dinamis berubah.
        self.turunan = {
            "ruas_terblokir": set(),
            "ruas_terbatas": set(),
            "rs_tidak_terjangkau": False,
            "jejak_inferensi": [],
            "p_macet_per_ruas": {},
            "biaya_per_ruas": {},
            "hasil_astar": None,
            "hasil_bfs": None,
            "hasil_dijkstra": None,
            "hasil_terpendek_jarak": None,
            "sudah_dihitung": False,  # penanda apakah turunan masih valid
        }

    # =========================================================================
    # METODE MEMUAT DATA TETAP
    # =========================================================================

    def muat_dari_file(self, path_graf: str, path_cpt: str):
        """
        Memuat pengetahuan tetap dari file JSON.

        Parameter:
            path_graf : path ke data/graf_contoh.json
            path_cpt  : path ke data/cpt_bawaan.json
        """
        # Muat graf
        with open(path_graf, "r", encoding="utf-8") as f:
            data_graf = json.load(f)
        self.tetap["simpul"] = data_graf["simpul"]
        self.tetap["ruas"] = data_graf["ruas"]

        # Muat CPT Bayesian
        with open(path_cpt, "r", encoding="utf-8") as f:
            data_cpt = json.load(f)
        self.tetap["cpt"] = data_cpt["cpt"]
        self.tetap["variabel_bayes"] = data_cpt["variabel"]

        # Reset turunan karena data tetap baru saja dimuat
        self._hapus_turunan()

    # =========================================================================
    # METODE MENGUBAH PENGETAHUAN DINAMIS
    # =========================================================================

    def tambah_fakta(self, predikat: str, objek: str):
        """
        Menambahkan fakta lapangan ke pengetahuan dinamis.
        Otomatis menghapus pengetahuan turunan yang sudah dihitung sebelumnya.

        Contoh: tambah_fakta("Kecelakaan", "B-RS")
        """
        self.dinamis["fakta"].add((predikat, objek))
        self._hapus_turunan()

    def hapus_fakta(self, predikat: str, objek: str):
        """
        Menghapus fakta lapangan dari pengetahuan dinamis.
        Otomatis menghapus pengetahuan turunan.

        Contoh: hapus_fakta("Kecelakaan", "B-RS")
        """
        self.dinamis["fakta"].discard((predikat, objek))
        self._hapus_turunan()

    def atur_konteks(self, jam: str, cuaca: str, hari_pasar: str):
        """
        Mengatur konteks lingkungan (jam, cuaca, hari pasar).
        Otomatis menghapus pengetahuan turunan.

        Parameter:
            jam        : "Sibuk" atau "Normal"
            cuaca      : "Hujan" atau "Cerah"
            hari_pasar : "Ya" atau "Tidak"
        """
        self.dinamis["konteks"]["jam"] = jam
        self.dinamis["konteks"]["cuaca"] = cuaca
        self.dinamis["konteks"]["hari_pasar"] = hari_pasar
        self._hapus_turunan()

    def atur_titik(self, titik_awal: str, tujuan: str):
        """
        Mengatur titik awal perjalanan dan RS tujuan.
        Otomatis menghapus pengetahuan turunan.
        """
        self.dinamis["titik_awal"] = titik_awal
        self.dinamis["tujuan"] = tujuan
        self._hapus_turunan()

    def atur_p_macet_manual(self, dict_ruas_ke_p: dict):
        """
        Mengatur P(macet) secara manual per ruas untuk mode validasi.
        Jika diisi, Bayesian Network diabaikan dan nilai manual yang dipakai.
        Kosongkan dengan dict {} untuk kembali ke mode Bayesian Network.

        Parameter:
            dict_ruas_ke_p : dict berisi {"id_ruas": float}, misal {"A-B": 0.80}
        """
        self.dinamis["p_macet_manual"] = dict_ruas_ke_p
        self._hapus_turunan()

    def reset_dinamis(self):
        """
        Menghapus semua fakta dinamis dan pengetahuan turunan.
        Pengetahuan tetap tidak terpengaruh.
        """
        self.dinamis["fakta"] = set()
        self.dinamis["konteks"] = {"jam": None, "cuaca": None, "hari_pasar": None}
        self.dinamis["titik_awal"] = None
        self.dinamis["tujuan"] = None
        self.dinamis["p_macet_manual"] = {}
        self._hapus_turunan()

    # =========================================================================
    # METODE MENYIMPAN PENGETAHUAN TURUNAN
    # =========================================================================

    def simpan_turunan(
        self,
        ruas_terblokir: set,
        ruas_terbatas: set,
        rs_tidak_terjangkau: bool,
        jejak_inferensi: list,
        p_macet_per_ruas: dict,
        biaya_per_ruas: dict,
        hasil_astar: dict,
        hasil_bfs: dict,
        hasil_dijkstra: dict,
        hasil_terpendek_jarak: dict,
    ):
        """
        Menyimpan hasil komputasi pipeline ke pengetahuan turunan.
        Dipanggil oleh pipeline.py setelah semua metode AI selesai berjalan.
        """
        self.turunan["ruas_terblokir"] = ruas_terblokir
        self.turunan["ruas_terbatas"] = ruas_terbatas
        self.turunan["rs_tidak_terjangkau"] = rs_tidak_terjangkau
        self.turunan["jejak_inferensi"] = jejak_inferensi
        self.turunan["p_macet_per_ruas"] = p_macet_per_ruas
        self.turunan["biaya_per_ruas"] = biaya_per_ruas
        self.turunan["hasil_astar"] = hasil_astar
        self.turunan["hasil_bfs"] = hasil_bfs
        self.turunan["hasil_dijkstra"] = hasil_dijkstra
        self.turunan["hasil_terpendek_jarak"] = hasil_terpendek_jarak
        self.turunan["sudah_dihitung"] = True

    # =========================================================================
    # METODE INFORMASI DAN EKSPOR
    # =========================================================================

    def ringkasan(self) -> str:
        """
        Mengembalikan teks ringkas isi basis pengetahuan.
        Berguna untuk ditampilkan di antarmuka atau di log.
        """
        jml_simpul = len(self.tetap["simpul"])
        jml_ruas = len(self.tetap["ruas"])
        jml_aturan = len(self.tetap["aturan_fol"])
        jml_cpt = len(self.tetap["cpt"])
        jml_fakta = len(self.dinamis["fakta"])
        konteks = self.dinamis["konteks"]
        jam = konteks["jam"] or "-"
        cuaca = konteks["cuaca"] or "-"
        pasar = konteks["hari_pasar"] or "-"
        sudah = "Ya" if self.turunan["sudah_dihitung"] else "Belum"

        return (
            f"Graf: {jml_simpul} simpul, {jml_ruas} ruas | "
            f"Aturan FOL: {jml_aturan} | CPT: {jml_cpt} entri | "
            f"Fakta aktif: {jml_fakta} | "
            f"Konteks: {jam}/{cuaca}/pasar={pasar} | "
            f"Turunan dihitung: {sudah}"
        )

    def ekspor_markdown(self, path: str):
        """
        Mengekspor seluruh isi basis pengetahuan ke file Markdown.
        File ini dapat dipakai langsung sebagai bahan laporan.

        Parameter:
            path : path tujuan, misal "dokumen/basis_pengetahuan.md"
        """
        baris = []
        baris.append("# Basis Pengetahuan AmbuRoute\n")
        baris.append(
            "> File ini diekspor otomatis oleh sistem. "
            "Jangan diedit manual.\n"
        )
        baris.append(f"Diekspor pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # --- PENGETAHUAN TETAP ---
        baris.append("\n## 1. Pengetahuan Tetap\n")
        baris.append("Data yang dimuat sekali dari file dan tidak berubah saat sistem berjalan.\n")

        # Tabel simpul
        baris.append("\n### Simpul Graf\n")
        baris.append("| ID | Nama | Koordinat X | Koordinat Y |")
        baris.append("|---|---|---|---|")
        for s in self.tetap["simpul"]:
            baris.append(
                f"| {s['id']} | {s['nama']} "
                f"| {s['koordinat']['x']} | {s['koordinat']['y']} |"
            )

        # Tabel ruas
        baris.append("\n### Ruas Graf\n")
        baris.append("| ID | Dari | Ke | Jarak (km) | Tipe | Kecepatan (km/j) |")
        baris.append("|---|---|---|---|---|---|")
        for r in self.tetap["ruas"]:
            baris.append(
                f"| {r['id']} | {r['dari']} | {r['ke']} "
                f"| {r['jarak_km']} | {r['tipe']} | {r['kecepatan_kmh']} |"
            )

        # Aturan FOL
        baris.append("\n### Aturan First Order Logic (R1–R7)\n")
        baris.append("| Kode | Notasi FOL | Keterangan |")
        baris.append("|---|---|---|")
        for aturan in self.tetap["aturan_fol"]:
            baris.append(
                f"| {aturan['kode']} | `{aturan['notasi']}` | {aturan['keterangan']} |"
            )

        # CPT Bayesian
        baris.append("\n### CPT Bayesian Network (ASUMSI)\n")
        baris.append("| Jam | Cuaca | Hari Pasar | Tipe Ruas | P(Macet=Ya) |")
        baris.append("|---|---|---|---|---|")
        for entri in self.tetap["cpt"]:
            asumsi = " *(ASUMSI)*" if entri.get("_asumsi") else ""
            baris.append(
                f"| {entri['jam']} | {entri['cuaca']} | {entri['hari_pasar']} "
                f"| {entri['tipe_ruas']} | {entri['p_macet_ya']}{asumsi} |"
            )

        # --- PENGETAHUAN DINAMIS ---
        baris.append("\n## 2. Pengetahuan Dinamis\n")
        baris.append("Data yang berubah mengikuti masukan pengguna.\n")

        konteks = self.dinamis["konteks"]
        baris.append(f"- **Jam:** {konteks['jam'] or '(belum diatur)'}")
        baris.append(f"- **Cuaca:** {konteks['cuaca'] or '(belum diatur)'}")
        baris.append(f"- **Hari Pasar:** {konteks['hari_pasar'] or '(belum diatur)'}")
        baris.append(f"- **Titik Awal:** {self.dinamis['titik_awal'] or '(belum diatur)'}")
        baris.append(f"- **Tujuan:** {self.dinamis['tujuan'] or '(belum diatur)'}")

        baris.append("\n**Fakta Lapangan Aktif:**\n")
        if self.dinamis["fakta"]:
            for predikat, objek in sorted(self.dinamis["fakta"]):
                baris.append(f"- {predikat}({objek})")
        else:
            baris.append("- *(tidak ada fakta aktif)*")

        if self.dinamis["p_macet_manual"]:
            baris.append("\n**Mode Validasi Manual — P(macet) per Ruas:**\n")
            baris.append("| Ruas | P(Macet) |")
            baris.append("|---|---|")
            for ruas_id, p in self.dinamis["p_macet_manual"].items():
                baris.append(f"| {ruas_id} | {p} |")

        # --- PENGETAHUAN TURUNAN ---
        baris.append("\n## 3. Pengetahuan Turunan\n")
        baris.append("Hasil komputasi sistem. Dihapus otomatis saat fakta dinamis berubah.\n")

        if not self.turunan["sudah_dihitung"]:
            baris.append("*(belum dihitung — jalankan pipeline terlebih dahulu)*\n")
        else:
            terblokir = self.turunan["ruas_terblokir"]
            terbatas = self.turunan["ruas_terbatas"]
            baris.append(f"- **Ruas Terblokir:** {', '.join(sorted(terblokir)) or '(tidak ada)'}")
            baris.append(f"- **Ruas Terbatas:** {', '.join(sorted(terbatas)) or '(tidak ada)'}")
            baris.append(f"- **RS Tidak Terjangkau:** {'Ya' if self.turunan['rs_tidak_terjangkau'] else 'Tidak'}")

            if self.turunan["p_macet_per_ruas"]:
                baris.append("\n**P(Macet) per Ruas:**\n")
                baris.append("| Ruas | P(Macet) |")
                baris.append("|---|---|")
                for ruas_id, p in sorted(self.turunan["p_macet_per_ruas"].items()):
                    baris.append(f"| {ruas_id} | {p:.4f} |")

            if self.turunan["jejak_inferensi"]:
                baris.append("\n**Jejak Inferensi FOL:**\n")
                for langkah in self.turunan["jejak_inferensi"]:
                    baris.append(f"- {langkah}")

            if self.turunan["hasil_astar"]:
                hasil = self.turunan["hasil_astar"]
                baris.append(f"\n**Hasil A*:** {' → '.join(hasil['rute'])} "
                             f"({hasil['total_waktu_menit']:.2f} menit, "
                             f"{hasil['total_jarak_km']:.1f} km)")

        # Tulis ke file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(baris) + "\n")

    # =========================================================================
    # METODE INTERNAL
    # =========================================================================

    def _hapus_turunan(self):
        """
        Menghapus semua pengetahuan turunan.
        Dipanggil otomatis setiap kali fakta atau konteks dinamis berubah,
        agar hasil komputasi lama tidak ikut dipakai pada pencarian berikutnya.
        """
        self.turunan = {
            "ruas_terblokir": set(),
            "ruas_terbatas": set(),
            "rs_tidak_terjangkau": False,
            "jejak_inferensi": [],
            "p_macet_per_ruas": {},
            "biaya_per_ruas": {},
            "hasil_astar": None,
            "hasil_bfs": None,
            "hasil_dijkstra": None,
            "hasil_terpendek_jarak": None,
            "sudah_dihitung": False,
        }

    @staticmethod
    def _daftar_aturan_fol() -> list:
        """
        Mengembalikan daftar aturan FOL R1-R7 beserta teks notasinya.
        Data ini bersifat tetap (tidak berubah) dan dipakai oleh laporan.
        """
        return [
            {
                "kode": "R1",
                "notasi": "∀r  Kecelakaan(r) ∧ Tertutup(r) → Terblokir(r)",
                "keterangan": "Kecelakaan yang menutup jalan → ruas terblokir",
            },
            {
                "kode": "R2",
                "notasi": "∀r  Banjir(r) → Terblokir(r)",
                "keterangan": "Banjir selalu memblokir ruas",
            },
            {
                "kode": "R3",
                "notasi": "∀r  Kecelakaan(r) ∧ ¬Tertutup(r) ∧ ¬LajurTersisaLebihDariSatu(r) → Terblokir(r)",
                "keterangan": "Kecelakaan, jalan tidak tertutup, hanya satu lajur → terblokir",
            },
            {
                "kode": "R4",
                "notasi": "∀r  Kecelakaan(r) ∧ ¬Tertutup(r) ∧ LajurTersisaLebihDariSatu(r) → Terbatas(r)",
                "keterangan": "Kecelakaan, jalan tidak tertutup, lebih dari satu lajur → terbatas",
            },
            {
                "kode": "R5",
                "notasi": "∀r  ¬Terblokir(r) → DapatDilalui(r)",
                "keterangan": "Ruas yang tidak terblokir dapat dilalui (asumsi dunia tertutup)",
            },
            {
                "kode": "R6",
                "notasi": "∀s  (∀r  Terhubung(s,r) → Terblokir(r)) → Terisolasi(s)",
                "keterangan": "Simpul yang semua ruasnya terblokir → terisolasi",
            },
            {
                "kode": "R7",
                "notasi": "Terisolasi(RS) ∨ Terisolasi(titik_awal) → RSTidakTerjangkau",
                "keterangan": "RS atau titik awal terisolasi → RS tidak terjangkau",
            },
        ]
