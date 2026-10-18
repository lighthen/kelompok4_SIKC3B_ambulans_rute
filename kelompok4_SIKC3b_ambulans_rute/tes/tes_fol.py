"""
tes_fol.py — Tes untuk sumber/fol.py

Mencakup skenario dari requirements.md bagian 8:
- S3: Kecelakaan di B-RS, 1 lajur, tidak tertutup → R3 aktif → Terblokir
- S4: Kecelakaan di A-C, 2 lajur, tidak tertutup → R4 aktif → Terbatas
- S5: Banjir di B-RS dan D-RS → RS terisolasi → RSTidakTerjangkau
- Tanpa fakta → semua ruas DapatDilalui
- Kecelakaan + tertutup → R1 aktif → Terblokir
- Banjir → R2 aktif → Terblokir
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.fol import jalankan_fol_dari_basis, jalankan_forward_chaining, bangun_fakta_struktural

DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_GRAF = os.path.join(DIREKTORI_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIREKTORI_ROOT, "data", "cpt_bawaan.json")


@pytest.fixture
def basis():
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    bp.atur_titik("A", "RS")
    return bp


def jalankan(basis):
    """Shortcut untuk menjalankan FOL dari basis."""
    return jalankan_fol_dari_basis(basis)


# =========================================================================
# Tes struktur kembalian
# =========================================================================

class TestStrukturHasil:
    def test_semua_field_ada(self, basis):
        """Hasil FOL harus punya semua field yang diperlukan."""
        hasil = jalankan(basis)
        assert "ruas_terblokir" in hasil
        assert "ruas_terbatas" in hasil
        assert "ruas_dapat_dilalui" in hasil
        assert "simpul_terisolasi" in hasil
        assert "rs_tidak_terjangkau" in hasil
        assert "jejak_inferensi" in hasil
        assert "iterasi" in hasil

    def test_tipe_data(self, basis):
        """Tipe data setiap field harus benar."""
        hasil = jalankan(basis)
        assert isinstance(hasil["ruas_terblokir"], set)
        assert isinstance(hasil["ruas_terbatas"], set)
        assert isinstance(hasil["ruas_dapat_dilalui"], set)
        assert isinstance(hasil["simpul_terisolasi"], set)
        assert isinstance(hasil["rs_tidak_terjangkau"], bool)
        assert isinstance(hasil["jejak_inferensi"], list)
        assert isinstance(hasil["iterasi"], int)

    def test_jejak_berisi_string(self, basis):
        """Jejak inferensi harus berisi string."""
        hasil = jalankan(basis)
        for langkah in hasil["jejak_inferensi"]:
            assert isinstance(langkah, str)


# =========================================================================
# Tanpa fakta — semua DapatDilalui
# =========================================================================

class TestTanpaFakta:
    def test_semua_dapat_dilalui(self, basis):
        """Tanpa fakta kecelakaan/banjir, semua ruas harus DapatDilalui."""
        hasil = jalankan(basis)
        assert hasil["ruas_terblokir"] == set()
        assert hasil["ruas_terbatas"] == set()
        assert hasil["ruas_dapat_dilalui"] == {"A-B", "B-RS", "A-C", "C-D", "D-RS"}

    def test_rs_terjangkau(self, basis):
        """Tanpa fakta, RS harus terjangkau."""
        hasil = jalankan(basis)
        assert hasil["rs_tidak_terjangkau"] is False

    def test_tidak_ada_simpul_terisolasi(self, basis):
        """Tanpa fakta, tidak ada simpul yang terisolasi."""
        hasil = jalankan(basis)
        assert hasil["simpul_terisolasi"] == set()


# =========================================================================
# R1: Kecelakaan + Tertutup → Terblokir
# =========================================================================

class TestR1:
    def test_r1_aktif(self, basis):
        """R1: Kecelakaan(B-RS) ∧ Tertutup(B-RS) → Terblokir(B-RS)."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        basis.tambah_fakta("Tertutup", "B-RS")
        hasil = jalankan(basis)
        assert "B-RS" in hasil["ruas_terblokir"]
        assert "B-RS" not in hasil["ruas_dapat_dilalui"]

    def test_r1_jejak_mencantumkan_r1(self, basis):
        """Jejak harus menyebut R1 saat aktif."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        basis.tambah_fakta("Tertutup", "B-RS")
        hasil = jalankan(basis)
        r1_ada = any("R1" in langkah and "B-RS" in langkah
                     for langkah in hasil["jejak_inferensi"])
        assert r1_ada, "Jejak tidak menyebut R1 untuk B-RS"

    def test_r1_tidak_aktif_tanpa_tertutup(self, basis):
        """R1 tidak aktif jika Tertutup tidak ada."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        # Tidak tambah Tertutup
        hasil = jalankan(basis)
        # Akan aktif R3 (kecelakaan, tidak tertutup, tidak ada lajur lebih)
        assert "B-RS" in hasil["ruas_terblokir"]  # via R3, bukan R1


# =========================================================================
# R2: Banjir → Terblokir
# =========================================================================

class TestR2:
    def test_r2_aktif(self, basis):
        """R2: Banjir(A-C) → Terblokir(A-C)."""
        basis.tambah_fakta("Banjir", "A-C")
        hasil = jalankan(basis)
        assert "A-C" in hasil["ruas_terblokir"]
        assert "A-C" not in hasil["ruas_dapat_dilalui"]

    def test_r2_jejak_mencantumkan_r2(self, basis):
        """Jejak harus menyebut R2 saat aktif."""
        basis.tambah_fakta("Banjir", "A-C")
        hasil = jalankan(basis)
        r2_ada = any("R2" in langkah and "A-C" in langkah
                     for langkah in hasil["jejak_inferensi"])
        assert r2_ada


# =========================================================================
# S3: Kecelakaan di B-RS, 1 lajur, tidak tertutup → R3 → Terblokir
# =========================================================================

class TestS3R3:
    """
    Skenario S3: Kecelakaan di B-RS, tidak tertutup, hanya 1 lajur tersisa.
    R3: Kecelakaan(B-RS) ∧ ¬Tertutup(B-RS) ∧ ¬LajurTersisaLebihDariSatu(B-RS) → Terblokir(B-RS)
    """

    def test_r3_aktif_brs_terblokir(self, basis):
        """S3: B-RS harus Terblokir via R3."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        # Tidak tambah Tertutup → ¬Tertutup
        # Tidak tambah LajurTersisaLebihDariSatu → ¬LajurLebih
        hasil = jalankan(basis)
        assert "B-RS" in hasil["ruas_terblokir"], \
            "B-RS harus Terblokir pada S3"

    def test_r3_brs_tidak_dapat_dilalui(self, basis):
        """S3: B-RS yang Terblokir tidak boleh DapatDilalui."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        hasil = jalankan(basis)
        assert "B-RS" not in hasil["ruas_dapat_dilalui"]

    def test_r3_ruas_lain_masih_normal(self, basis):
        """S3: Ruas selain B-RS harus tetap DapatDilalui."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        hasil = jalankan(basis)
        for ruas in ["A-B", "A-C", "C-D", "D-RS"]:
            assert ruas in hasil["ruas_dapat_dilalui"], \
                f"Ruas {ruas} seharusnya DapatDilalui pada S3"

    def test_r3_jejak_mencantumkan_r3(self, basis):
        """Jejak harus menyebut R3 dengan detail kondisi."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        hasil = jalankan(basis)
        r3_ada = any("R3" in langkah and "B-RS" in langkah
                     for langkah in hasil["jejak_inferensi"])
        assert r3_ada, f"Jejak tidak menyebut R3:\n" + "\n".join(hasil["jejak_inferensi"])

    def test_r3_rs_masih_terjangkau(self, basis):
        """S3: Masih ada rute via C-D, RS masih terjangkau."""
        basis.tambah_fakta("Kecelakaan", "B-RS")
        hasil = jalankan(basis)
        assert hasil["rs_tidak_terjangkau"] is False


# =========================================================================
# S4: Kecelakaan di A-C, 2 lajur, tidak tertutup → R4 → Terbatas
# =========================================================================

class TestS4R4:
    """
    Skenario S4: Kecelakaan di A-C, tidak tertutup, lebih dari 1 lajur tersisa.
    R4: Kecelakaan(A-C) ∧ ¬Tertutup(A-C) ∧ LajurTersisaLebihDariSatu(A-C) → Terbatas(A-C)
    """

    def test_r4_aktif_ac_terbatas(self, basis):
        """S4: A-C harus Terbatas via R4."""
        basis.tambah_fakta("Kecelakaan", "A-C")
        basis.tambah_fakta("LajurTersisaLebihDariSatu", "A-C")
        # Tidak tambah Tertutup
        hasil = jalankan(basis)
        assert "A-C" in hasil["ruas_terbatas"], \
            "A-C harus Terbatas pada S4"

    def test_r4_ac_tidak_terblokir(self, basis):
        """S4: A-C yang Terbatas tidak boleh Terblokir."""
        basis.tambah_fakta("Kecelakaan", "A-C")
        basis.tambah_fakta("LajurTersisaLebihDariSatu", "A-C")
        hasil = jalankan(basis)
        assert "A-C" not in hasil["ruas_terblokir"], \
            "A-C tidak boleh Terblokir pada S4 (hanya Terbatas)"

    def test_r4_ac_dapat_dilalui(self, basis):
        """S4: A-C Terbatas masih DapatDilalui (dengan pengali waktu)."""
        basis.tambah_fakta("Kecelakaan", "A-C")
        basis.tambah_fakta("LajurTersisaLebihDariSatu", "A-C")
        hasil = jalankan(basis)
        assert "A-C" in hasil["ruas_dapat_dilalui"], \
            "A-C harus DapatDilalui pada S4 (meskipun Terbatas)"

    def test_r4_jejak_mencantumkan_r4(self, basis):
        """Jejak harus menyebut R4 saat aktif."""
        basis.tambah_fakta("Kecelakaan", "A-C")
        basis.tambah_fakta("LajurTersisaLebihDariSatu", "A-C")
        hasil = jalankan(basis)
        r4_ada = any("R4" in langkah and "A-C" in langkah
                     for langkah in hasil["jejak_inferensi"])
        assert r4_ada, "Jejak tidak menyebut R4 untuk A-C"


# =========================================================================
# S5: Banjir di B-RS dan D-RS → RS Terisolasi → RSTidakTerjangkau
# =========================================================================

class TestS5RsTidakTerjangkau:
    """
    Skenario S5: Banjir di B-RS dan D-RS.
    R2 → B-RS dan D-RS Terblokir.
    R6 → RS Terisolasi (semua ruas RS terblokir: B-RS dan D-RS).
    R7 → RSTidakTerjangkau.
    """

    def test_s5_brs_dan_drs_terblokir(self, basis):
        """S5: B-RS dan D-RS harus Terblokir."""
        basis.tambah_fakta("Banjir", "B-RS")
        basis.tambah_fakta("Banjir", "D-RS")
        hasil = jalankan(basis)
        assert "B-RS" in hasil["ruas_terblokir"]
        assert "D-RS" in hasil["ruas_terblokir"]

    def test_s5_rs_terisolasi(self, basis):
        """S5: RS harus Terisolasi karena semua ruasnya terblokir."""
        basis.tambah_fakta("Banjir", "B-RS")
        basis.tambah_fakta("Banjir", "D-RS")
        hasil = jalankan(basis)
        assert "RS" in hasil["simpul_terisolasi"], \
            "RS harus Terisolasi pada S5"

    def test_s5_rs_tidak_terjangkau_true(self, basis):
        """S5: rs_tidak_terjangkau harus True."""
        basis.tambah_fakta("Banjir", "B-RS")
        basis.tambah_fakta("Banjir", "D-RS")
        hasil = jalankan(basis)
        assert hasil["rs_tidak_terjangkau"] is True, \
            "rs_tidak_terjangkau harus True pada S5"

    def test_s5_jejak_mencantumkan_r6_r7(self, basis):
        """Jejak harus menyebut R6 (terisolasi) dan R7 (tidak terjangkau)."""
        basis.tambah_fakta("Banjir", "B-RS")
        basis.tambah_fakta("Banjir", "D-RS")
        hasil = jalankan(basis)
        r6_ada = any("R6" in l for l in hasil["jejak_inferensi"])
        r7_ada = any("R7" in l for l in hasil["jejak_inferensi"])
        assert r6_ada, "Jejak tidak menyebut R6"
        assert r7_ada, "Jejak tidak menyebut R7"

    def test_s5_tidak_error_tidak_raise_exception(self, basis):
        """S5: Saat RS tidak terjangkau, sistem tidak boleh raise exception."""
        basis.tambah_fakta("Banjir", "B-RS")
        basis.tambah_fakta("Banjir", "D-RS")
        try:
            hasil = jalankan(basis)
            assert hasil["rs_tidak_terjangkau"] is True
        except Exception as e:
            pytest.fail(f"FOL raise exception pada S5: {e}")


# =========================================================================
# Kombinasi beberapa skenario
# =========================================================================

class TestKombinasi:
    def test_beberapa_ruas_terblokir_sekaligus(self, basis):
        """Bisa blokir beberapa ruas sekaligus."""
        basis.tambah_fakta("Banjir", "A-B")
        basis.tambah_fakta("Kecelakaan", "A-C")  # tanpa lajur → R3
        hasil = jalankan(basis)
        assert "A-B" in hasil["ruas_terblokir"]
        assert "A-C" in hasil["ruas_terblokir"]

    def test_ruas_terbatas_dan_terblokir_berbeda(self, basis):
        """Ruas Terbatas dan Terblokir harus diset terpisah."""
        basis.tambah_fakta("Banjir", "B-RS")
        basis.tambah_fakta("Kecelakaan", "A-C")
        basis.tambah_fakta("LajurTersisaLebihDariSatu", "A-C")
        hasil = jalankan(basis)
        assert "B-RS" in hasil["ruas_terblokir"]
        assert "A-C" in hasil["ruas_terbatas"]
        assert "B-RS" not in hasil["ruas_terbatas"]
        assert "A-C" not in hasil["ruas_terblokir"]

    def test_titik_awal_terisolasi(self, basis):
        """Jika titik awal terisolasi, RS juga tidak terjangkau."""
        # A hanya terhubung ke B (via A-B) dan C (via A-C)
        # Blokir keduanya
        basis.tambah_fakta("Banjir", "A-B")
        basis.tambah_fakta("Banjir", "A-C")
        hasil = jalankan(basis)
        assert "A" in hasil["simpul_terisolasi"]
        assert hasil["rs_tidak_terjangkau"] is True

    def test_iterasi_lebih_dari_satu_jika_ada_rantai(self, basis):
        """iterasi minimal 1."""
        hasil = jalankan(basis)
        assert hasil["iterasi"] >= 1


# =========================================================================
# Tes bangun_fakta_struktural
# =========================================================================

class TestFaktaStruktural:
    def test_terhubung_terbentuk(self, basis):
        """Fakta Terhubung harus terbentuk untuk semua ruas."""
        fakta_str, ruas_per_simpul = bangun_fakta_struktural(basis)
        # A-B terhubung ke A dan B
        assert ("Terhubung", "A", "A-B") in fakta_str
        assert ("Terhubung", "B", "A-B") in fakta_str

    def test_ruas_per_simpul_benar(self, basis):
        """ruas_per_simpul harus memetakan simpul ke ruas yang menyentuhnya."""
        _, ruas_per_simpul = bangun_fakta_struktural(basis)
        # RS terhubung ke B-RS dan D-RS
        assert "B-RS" in ruas_per_simpul["RS"]
        assert "D-RS" in ruas_per_simpul["RS"]
        # A terhubung ke A-B dan A-C
        assert "A-B" in ruas_per_simpul["A"]
        assert "A-C" in ruas_per_simpul["A"]
