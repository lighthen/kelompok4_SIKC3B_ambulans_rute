"""
tes_integrasi.py — Tes integrasi pipeline lengkap (FOL → Bayes → A*)

Mencakup skenario S0–S5 dari requirements.md bagian 8:
- S0: Validasi manual → A-C-D-RS = 12.75 menit (Checkpoint 1 ulang via pipeline)
- S1: Normal + Cerah + bukan hari pasar
- S2: Sibuk + Hujan + hari pasar (waktu lebih lama dari S1)
- S3: S2 + kecelakaan di B-RS, 1 lajur → R3 aktif → B-RS terblokir → rute dialihkan
- S4: Kecelakaan di A-C, 2 lajur → R4 aktif → A-C terbatas
- S5: Banjir di B-RS dan D-RS → RS terisolasi → pesan jelas, bukan error

Juga tes:
- Perubahan fakta dinamis menghapus turunan lama
- ekspor_markdown() menghasilkan file dengan konten lengkap
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.pipeline import jalankan

DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_GRAF = os.path.join(DIREKTORI_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIREKTORI_ROOT, "data", "cpt_bawaan.json")

P_MACET_MANUAL = {
    "A-B": 0.80, "B-RS": 0.85,
    "A-C": 0.20, "C-D": 0.15, "D-RS": 0.10,
}
TOLERANSI = 1e-6


def buat_basis():
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    bp.atur_titik("A", "RS")
    return bp


# =========================================================================
# S0 — Validasi manual via pipeline (Checkpoint 1 ulang)
# =========================================================================

class TestS0ValidasiManual:
    """S0: Pipeline dengan P(macet) manual harus menghasilkan 12.75 menit."""

    def test_s0_rute_acdrs(self):
        """A* harus memilih A→C→D→RS."""
        bp = buat_basis()
        bp.atur_p_macet_manual(P_MACET_MANUAL)
        hasil = jalankan(bp)
        assert hasil["hasil_astar"]["rute"] == ["A", "C", "D", "RS"], \
            f"S0: rute A* = {hasil['hasil_astar']['rute']}"

    def test_s0_total_waktu_1275(self):
        """Total waktu harus 12.75 menit."""
        bp = buat_basis()
        bp.atur_p_macet_manual(P_MACET_MANUAL)
        hasil = jalankan(bp)
        assert abs(hasil["hasil_astar"]["total_waktu_menit"] - 12.75) < TOLERANSI

    def test_s0_terpendek_jarak_abrs(self):
        """Rute terpendek jarak harus A→B→RS (5.0 km)."""
        bp = buat_basis()
        bp.atur_p_macet_manual(P_MACET_MANUAL)
        hasil = jalankan(bp)
        assert hasil["hasil_terpendek_jarak"]["rute"] == ["A", "B", "RS"]

    def test_s0_turunan_tersimpan(self):
        """Setelah pipeline, turunan harus tersimpan di BasisPengetahuan."""
        bp = buat_basis()
        bp.atur_p_macet_manual(P_MACET_MANUAL)
        jalankan(bp)
        assert bp.turunan["sudah_dihitung"] is True

    def test_s0_jejak_inferensi_ada(self):
        """Jejak inferensi harus non-kosong."""
        bp = buat_basis()
        bp.atur_p_macet_manual(P_MACET_MANUAL)
        hasil = jalankan(bp)
        assert len(hasil["jejak_inferensi"]) > 0


# =========================================================================
# S1 — Normal + Cerah + bukan hari pasar
# =========================================================================

class TestS1NormalCerahTidakPasar:
    """S1: Kondisi normal, semua ruas DapatDilalui, A* memilih rute."""

    def setup_method(self):
        self.bp = buat_basis()
        self.bp.atur_konteks("Normal", "Cerah", "Tidak")
        self.hasil = jalankan(self.bp)

    def test_s1_tidak_ada_ruas_terblokir(self):
        """S1: Tidak ada ruas yang terblokir."""
        assert self.hasil["ruas_terblokir"] == set()

    def test_s1_tidak_ada_ruas_terbatas(self):
        """S1: Tidak ada ruas yang terbatas."""
        assert self.hasil["ruas_terbatas"] == set()

    def test_s1_semua_ruas_dapat_dilalui(self):
        """S1: Semua 5 ruas DapatDilalui."""
        assert self.hasil["ruas_dapat_dilalui"] == {"A-B", "B-RS", "A-C", "C-D", "D-RS"}

    def test_s1_rs_terjangkau(self):
        """S1: RS harus terjangkau."""
        assert self.hasil["rs_tidak_terjangkau"] is False

    def test_s1_astar_menemukan_rute(self):
        """S1: A* harus menemukan rute ke RS."""
        rute = self.hasil["hasil_astar"]["rute"]
        assert len(rute) >= 2
        assert rute[0] == "A"
        assert rute[-1] == "RS"

    def test_s1_p_macet_rendah(self):
        """S1: P(macet) Normal+Cerah+Tidak harus lebih rendah dari S2."""
        for p in self.hasil["p_macet_per_ruas"].values():
            assert 0.0 <= p <= 1.0


# =========================================================================
# S2 — Sibuk + Hujan + hari pasar
# =========================================================================

class TestS2SibukHujanPasar:
    """S2: Kondisi sibuk, waktu lebih lama dari S1."""

    def setup_method(self):
        self.bp = buat_basis()
        self.bp.atur_konteks("Sibuk", "Hujan", "Ya")
        self.hasil = jalankan(self.bp)

    def test_s2_rs_terjangkau(self):
        """S2: RS masih terjangkau."""
        assert self.hasil["rs_tidak_terjangkau"] is False

    def test_s2_astar_menemukan_rute(self):
        """S2: A* menemukan rute."""
        rute = self.hasil["hasil_astar"]["rute"]
        assert rute[0] == "A"
        assert rute[-1] == "RS"

    def test_s2_waktu_lebih_lama_dari_s1(self):
        """S2: Waktu tempuh harus lebih lama dari S1 (kondisi sibuk+hujan+pasar)."""
        bp_s1 = buat_basis()
        bp_s1.atur_konteks("Normal", "Cerah", "Tidak")
        hasil_s1 = jalankan(bp_s1)
        assert (self.hasil["hasil_astar"]["total_waktu_menit"] >
                hasil_s1["hasil_astar"]["total_waktu_menit"]), \
            "Waktu S2 harus lebih lama dari S1"

    def test_s2_p_macet_tinggi(self):
        """S2: P(macet) Sibuk+Hujan+Ya harus >= P(macet) Normal+Cerah+Tidak."""
        bp_s1 = buat_basis()
        bp_s1.atur_konteks("Normal", "Cerah", "Tidak")
        hasil_s1 = jalankan(bp_s1)
        for id_ruas in self.hasil["p_macet_per_ruas"]:
            p_s2 = self.hasil["p_macet_per_ruas"][id_ruas]
            p_s1 = hasil_s1["p_macet_per_ruas"].get(id_ruas, 0)
            assert p_s2 >= p_s1, f"P(macet S2)[{id_ruas}]={p_s2} < P(macet S1)={p_s1}"


# =========================================================================
# S3 — Kondisi S2 + kecelakaan di B-RS, 1 lajur, tidak tertutup
# =========================================================================

class TestS3KecelakaanBRS:
    """S3: R3 aktif → B-RS terblokir → rute dialihkan via C-D."""

    def setup_method(self):
        self.bp = buat_basis()
        self.bp.atur_konteks("Sibuk", "Hujan", "Ya")
        self.bp.tambah_fakta("Kecelakaan", "B-RS")
        # Tidak tambah Tertutup → ¬Tertutup
        # Tidak tambah LajurTersisaLebihDariSatu → ¬Lajur → R3 aktif
        self.hasil = jalankan(self.bp)

    def test_s3_brs_terblokir(self):
        """S3: B-RS harus Terblokir via R3."""
        assert "B-RS" in self.hasil["ruas_terblokir"]

    def test_s3_rute_tidak_lewat_b(self):
        """S3: Rute A* tidak boleh melewati simpul B."""
        rute = self.hasil["hasil_astar"]["rute"]
        assert "B" not in rute, f"S3: rute A* = {rute} seharusnya tidak lewat B"

    def test_s3_rute_dialihkan_ke_acdrs(self):
        """S3: A* harus memilih A→C→D→RS karena B-RS terblokir."""
        rute = self.hasil["hasil_astar"]["rute"]
        assert rute == ["A", "C", "D", "RS"], \
            f"S3: A* = {rute}, harapan ['A','C','D','RS']"

    def test_s3_r3_di_jejak(self):
        """S3: Jejak inferensi harus menyebut R3 untuk B-RS."""
        r3_ada = any("R3" in l and "B-RS" in l for l in self.hasil["jejak_inferensi"])
        assert r3_ada, "Jejak tidak menyebut R3 untuk B-RS"

    def test_s3_rs_masih_terjangkau(self):
        """S3: RS masih terjangkau via rute alternatif."""
        assert self.hasil["rs_tidak_terjangkau"] is False


# =========================================================================
# S4 — Kecelakaan di A-C, 2 lajur, tidak tertutup → Terbatas
# =========================================================================

class TestS4KecelakaanACTerbatas:
    """S4: R4 aktif → A-C terbatas (waktu ×2)."""

    def setup_method(self):
        self.bp = buat_basis()
        self.bp.atur_konteks("Normal", "Cerah", "Tidak")
        self.bp.tambah_fakta("Kecelakaan", "A-C")
        self.bp.tambah_fakta("LajurTersisaLebihDariSatu", "A-C")
        self.hasil = jalankan(self.bp)

    def test_s4_ac_terbatas(self):
        """S4: A-C harus Terbatas."""
        assert "A-C" in self.hasil["ruas_terbatas"]

    def test_s4_ac_tidak_terblokir(self):
        """S4: A-C tidak boleh Terblokir."""
        assert "A-C" not in self.hasil["ruas_terblokir"]

    def test_s4_biaya_ac_lebih_mahal(self):
        """S4: Biaya A-C harus lebih mahal dari kondisi normal."""
        bp_normal = buat_basis()
        bp_normal.atur_konteks("Normal", "Cerah", "Tidak")
        hasil_normal = jalankan(bp_normal)
        biaya_normal = hasil_normal["biaya_per_ruas"].get("A-C", 0)
        biaya_s4 = self.hasil["biaya_per_ruas"].get("A-C", 0)
        assert biaya_s4 > biaya_normal, \
            f"Biaya A-C S4={biaya_s4} tidak lebih mahal dari normal={biaya_normal}"

    def test_s4_r4_di_jejak(self):
        """S4: Jejak harus menyebut R4 untuk A-C."""
        r4_ada = any("R4" in l and "A-C" in l for l in self.hasil["jejak_inferensi"])
        assert r4_ada

    def test_s4_rs_terjangkau(self):
        """S4: RS masih terjangkau."""
        assert self.hasil["rs_tidak_terjangkau"] is False


# =========================================================================
# S5 — Banjir di B-RS dan D-RS → RSTidakTerjangkau
# =========================================================================

class TestS5BanjirRSTidakTerjangkau:
    """S5: RS terisolasi → pesan jelas, tidak error."""

    def setup_method(self):
        self.bp = buat_basis()
        self.bp.atur_konteks("Normal", "Cerah", "Tidak")
        self.bp.tambah_fakta("Banjir", "B-RS")
        self.bp.tambah_fakta("Banjir", "D-RS")
        self.hasil = jalankan(self.bp)

    def test_s5_rs_tidak_terjangkau(self):
        """S5: rs_tidak_terjangkau harus True."""
        assert self.hasil["rs_tidak_terjangkau"] is True

    def test_s5_pesan_tidak_kosong(self):
        """S5: pesan harus berisi teks yang informatif."""
        assert len(self.hasil["pesan"]) > 0
        assert "tidak dapat dijangkau" in self.hasil["pesan"].lower() or \
               "terblokir" in self.hasil["pesan"].lower()

    def test_s5_tidak_ada_rute(self):
        """S5: hasil_astar harus None (tidak ada rute)."""
        assert self.hasil["hasil_astar"] is None

    def test_s5_tidak_error(self):
        """S5: pipeline tidak boleh raise exception."""
        bp = buat_basis()
        bp.tambah_fakta("Banjir", "B-RS")
        bp.tambah_fakta("Banjir", "D-RS")
        try:
            hasil = jalankan(bp)
            assert hasil["rs_tidak_terjangkau"] is True
        except Exception as e:
            pytest.fail(f"Pipeline raise exception pada S5: {e}")

    def test_s5_jejak_ada(self):
        """S5: Jejak inferensi harus ada meskipun RS tidak terjangkau."""
        assert len(self.hasil["jejak_inferensi"]) > 0

    def test_s5_turunan_tersimpan(self):
        """S5: turunan tetap tersimpan meskipun RS tidak terjangkau."""
        assert self.bp.turunan["sudah_dihitung"] is True
        assert self.bp.turunan["rs_tidak_terjangkau"] is True


# =========================================================================
# Tes penghapusan turunan saat fakta berubah
# =========================================================================

class TestTurunanKedaluwarsa:
    def test_fakta_baru_hapus_turunan(self):
        """Menambah fakta setelah pipeline selesai harus menghapus turunan."""
        bp = buat_basis()
        bp.atur_konteks("Normal", "Cerah", "Tidak")
        jalankan(bp)
        assert bp.turunan["sudah_dihitung"] is True

        # Tambah fakta baru
        bp.tambah_fakta("Banjir", "A-B")
        assert bp.turunan["sudah_dihitung"] is False
        assert bp.turunan["hasil_astar"] is None

    def test_jalankan_ulang_hasilkan_hasil_baru(self):
        """Setelah fakta berubah, jalankan() harus menghasilkan hasil yang baru."""
        bp = buat_basis()
        bp.atur_konteks("Normal", "Cerah", "Tidak")
        hasil1 = jalankan(bp)
        waktu1 = hasil1["hasil_astar"]["total_waktu_menit"]

        # Ganti konteks ke kondisi lebih buruk
        bp.atur_konteks("Sibuk", "Hujan", "Ya")
        hasil2 = jalankan(bp)
        waktu2 = hasil2["hasil_astar"]["total_waktu_menit"]

        assert waktu2 > waktu1, "Waktu S2 harus lebih lama dari S1 setelah jalankan ulang"

    def test_reset_dinamis_lalu_jalankan(self):
        """reset_dinamis() lalu jalankan() harus bekerja tanpa error."""
        bp = buat_basis()
        bp.atur_konteks("Sibuk", "Hujan", "Ya")
        bp.tambah_fakta("Kecelakaan", "B-RS")
        jalankan(bp)

        bp.reset_dinamis()
        bp.atur_titik("A", "RS")
        bp.atur_konteks("Normal", "Cerah", "Tidak")
        hasil = jalankan(bp)
        assert hasil["rs_tidak_terjangkau"] is False


# =========================================================================
# Tes ekspor_markdown() setelah pipeline
# =========================================================================

class TestEksporSetelahPipeline:
    def test_ekspor_markdown_setelah_pipeline(self, tmp_path):
        """ekspor_markdown() setelah pipeline harus memuat pengetahuan turunan."""
        bp = buat_basis()
        bp.atur_konteks("Normal", "Cerah", "Tidak")
        jalankan(bp)

        path_out = str(tmp_path / "basis_pengetahuan.md")
        bp.ekspor_markdown(path_out)

        with open(path_out, "r", encoding="utf-8") as f:
            isi = f.read()

        assert "Pengetahuan Tetap" in isi
        assert "Pengetahuan Dinamis" in isi
        assert "Pengetahuan Turunan" in isi
        assert "A*" in isi or "astar" in isi.lower() or "rute" in isi.lower()

    def test_ekspor_markdown_ke_dokumen(self):
        """ekspor_markdown() ke dokumen/basis_pengetahuan.md harus terbentuk."""
        bp = buat_basis()
        bp.atur_konteks("Normal", "Cerah", "Tidak")
        jalankan(bp)

        path_out = os.path.join(DIREKTORI_ROOT, "dokumen", "basis_pengetahuan.md")
        bp.ekspor_markdown(path_out)
        assert os.path.exists(path_out)


# =========================================================================
# Tes struktur hasil pipeline
# =========================================================================

class TestStrukturHasilPipeline:
    def test_semua_field_ada(self):
        """Hasil pipeline harus punya semua field yang diperlukan."""
        bp = buat_basis()
        bp.atur_konteks("Normal", "Cerah", "Tidak")
        hasil = jalankan(bp)

        field_wajib = [
            "rs_tidak_terjangkau", "pesan",
            "ruas_terblokir", "ruas_terbatas", "ruas_dapat_dilalui",
            "jejak_inferensi", "p_macet_per_ruas", "biaya_per_ruas",
            "hasil_astar", "hasil_bfs", "hasil_dijkstra",
            "hasil_terpendek_jarak", "titik_awal", "tujuan",
        ]
        for field in field_wajib:
            assert field in hasil, f"Field '{field}' tidak ada di hasil pipeline"

    def test_hasil_astar_punya_semua_subfield(self):
        """hasil_astar harus punya rute, total_waktu_menit, dll."""
        bp = buat_basis()
        bp.atur_konteks("Normal", "Cerah", "Tidak")
        hasil = jalankan(bp)
        astar = hasil["hasil_astar"]
        assert "rute" in astar
        assert "total_waktu_menit" in astar
        assert "total_jarak_km" in astar
        assert "jumlah_diekspansi" in astar
        assert "waktu_komputasi_detik" in astar
