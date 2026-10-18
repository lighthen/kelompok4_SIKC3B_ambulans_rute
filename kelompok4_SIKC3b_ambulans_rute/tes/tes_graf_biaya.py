"""
Tes untuk sumber/graf.py dan sumber/biaya.py

Mencakup:
- buat_graf() menghasilkan graf yang benar
- jarak_lurus() tepat secara geometri
- hitung_waktu_lancar() sesuai rumus
- hitung_waktu_harapan() sesuai tabel validasi manual (requirements.md bagian 6)
- hitung_biaya_ruas() dan hitung_semua_biaya() benar
- Konstanta FAKTOR_MACET dan FAKTOR_TERBATAS ada dan bernilai 2
"""
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.graf import buat_graf, jarak_lurus, ambil_id_ruas, kecepatan_maksimum
from sumber.biaya import (
    FAKTOR_MACET,
    FAKTOR_TERBATAS,
    hitung_waktu_lancar,
    hitung_waktu_harapan,
    hitung_biaya_ruas,
    hitung_semua_biaya,
)

DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_GRAF = os.path.join(DIREKTORI_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIREKTORI_ROOT, "data", "cpt_bawaan.json")


@pytest.fixture
def basis():
    """BasisPengetahuan yang sudah dimuat."""
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    return bp


@pytest.fixture
def graf(basis):
    """Graf networkx dari BasisPengetahuan."""
    return buat_graf(basis)


# =========================================================================
# Tes konstanta biaya.py
# =========================================================================

class TestKonstanta:
    def test_faktor_macet_adalah_dua(self):
        """FAKTOR_MACET harus bernilai 2 sesuai requirements.md."""
        assert FAKTOR_MACET == 2.0

    def test_faktor_terbatas_adalah_dua(self):
        """FAKTOR_TERBATAS harus bernilai 2 sesuai requirements.md."""
        assert FAKTOR_TERBATAS == 2.0


# =========================================================================
# Tes buat_graf()
# =========================================================================

class TestBuatGraf:
    def test_jumlah_simpul(self, graf):
        """Graf harus punya 5 simpul."""
        assert graf.number_of_nodes() == 5

    def test_jumlah_ruas(self, graf):
        """Graf harus punya 5 ruas."""
        assert graf.number_of_edges() == 5

    def test_simpul_ada(self, graf):
        """Semua ID simpul harus ada di graf."""
        for sid in ["A", "B", "C", "D", "RS"]:
            assert sid in graf.nodes, f"Simpul {sid} tidak ada di graf"

    def test_ruas_ada(self, graf):
        """Semua pasang simpul harus terhubung."""
        pasang = [("A", "B"), ("B", "RS"), ("A", "C"), ("C", "D"), ("D", "RS")]
        for dari, ke in pasang:
            assert graf.has_edge(dari, ke) or graf.has_edge(ke, dari), \
                f"Ruas {dari}-{ke} tidak ada di graf"

    def test_atribut_simpul(self, graf):
        """Setiap simpul harus punya atribut x, y, nama."""
        for sid in graf.nodes:
            assert "x" in graf.nodes[sid], f"Simpul {sid} tidak punya x"
            assert "y" in graf.nodes[sid], f"Simpul {sid} tidak punya y"
            assert "nama" in graf.nodes[sid], f"Simpul {sid} tidak punya nama"

    def test_atribut_ruas(self, graf):
        """Setiap ruas harus punya jarak_km, tipe, kecepatan_kmh, waktu_lancar."""
        for u, v, data in graf.edges(data=True):
            assert "jarak_km" in data, f"Ruas {u}-{v} tidak punya jarak_km"
            assert "tipe" in data, f"Ruas {u}-{v} tidak punya tipe"
            assert "kecepatan_kmh" in data, f"Ruas {u}-{v} tidak punya kecepatan_kmh"
            assert "waktu_lancar" in data, f"Ruas {u}-{v} tidak punya waktu_lancar"

    def test_waktu_lancar_ruas_ab(self, graf):
        """Waktu lancar A-B: 2.0 km / 40 km/j * 60 = 3.0 menit."""
        data = graf.get_edge_data("A", "B") or graf.get_edge_data("B", "A")
        assert abs(data["waktu_lancar"] - 3.0) < 1e-9

    def test_waktu_lancar_ruas_brs(self, graf):
        """Waktu lancar B-RS: 3.0 km / 40 km/j * 60 = 4.5 menit."""
        data = graf.get_edge_data("B", "RS") or graf.get_edge_data("RS", "B")
        assert abs(data["waktu_lancar"] - 4.5) < 1e-9

    def test_kecepatan_maksimum(self, graf):
        """Kecepatan maksimum harus 40 km/j."""
        assert kecepatan_maksimum(graf) == 40.0


# =========================================================================
# Tes jarak_lurus()
# =========================================================================

class TestJarakLurus:
    def test_jarak_a_ke_rs(self, graf):
        """Jarak lurus A(0,0) ke RS(5,0) = 5.0 km."""
        assert abs(jarak_lurus(graf, "A", "RS") - 5.0) < 1e-9

    def test_jarak_b_ke_rs(self, graf):
        """Jarak lurus B(2,0) ke RS(5,0) = 3.0 km."""
        assert abs(jarak_lurus(graf, "B", "RS") - 3.0) < 1e-9

    def test_jarak_d_ke_rs(self, graf):
        """Jarak lurus D(3.5,-1.0) ke RS(5,0) = sqrt(1.5^2+1^2) = sqrt(3.25) ≈ 1.803 km."""
        harapan = math.sqrt(1.5**2 + 1.0**2)
        assert abs(jarak_lurus(graf, "D", "RS") - harapan) < 1e-9

    def test_jarak_simetris(self, graf):
        """jarak_lurus(A,RS) harus sama dengan jarak_lurus(RS,A)."""
        assert abs(jarak_lurus(graf, "A", "RS") - jarak_lurus(graf, "RS", "A")) < 1e-9

    def test_jarak_ke_diri_sendiri_nol(self, graf):
        """Jarak simpul ke dirinya sendiri harus 0."""
        assert jarak_lurus(graf, "A", "A") == 0.0


# =========================================================================
# Tes ambil_id_ruas()
# =========================================================================

class TestAmbilIdRuas:
    def test_ambil_id_ab(self, graf):
        assert ambil_id_ruas(graf, "A", "B") == "A-B"

    def test_ambil_id_ba(self, graf):
        """Graf tak berarah — urutan boleh terbalik."""
        assert ambil_id_ruas(graf, "B", "A") == "A-B"

    def test_ruas_tidak_ada_error(self, graf):
        """Ruas yang tidak ada harus raise ValueError."""
        with pytest.raises(ValueError):
            ambil_id_ruas(graf, "A", "RS")


# =========================================================================
# Tes hitung_waktu_lancar()
# =========================================================================

class TestHitungWaktuLancar:
    def test_ruas_ab(self):
        """2.0 km / 40 km/j * 60 = 3.0 menit."""
        assert abs(hitung_waktu_lancar(2.0, 40) - 3.0) < 1e-9

    def test_ruas_brs(self):
        """3.0 km / 40 km/j * 60 = 4.5 menit."""
        assert abs(hitung_waktu_lancar(3.0, 40) - 4.5) < 1e-9

    def test_ruas_ac(self):
        """2.5 km / 40 km/j * 60 = 3.75 menit."""
        assert abs(hitung_waktu_lancar(2.5, 40) - 3.75) < 1e-9

    def test_ruas_cd(self):
        """2.0 km / 40 km/j * 60 = 3.0 menit."""
        assert abs(hitung_waktu_lancar(2.0, 40) - 3.0) < 1e-9

    def test_ruas_drs(self):
        """2.0 km / 40 km/j * 60 = 3.0 menit."""
        assert abs(hitung_waktu_lancar(2.0, 40) - 3.0) < 1e-9

    def test_kecepatan_nol_error(self):
        """Kecepatan 0 harus raise ValueError."""
        with pytest.raises(ValueError):
            hitung_waktu_lancar(2.0, 0)


# =========================================================================
# Tes hitung_waktu_harapan() — verifikasi tabel requirements.md bagian 6
# =========================================================================

class TestHitungWaktuHarapan:
    """
    Verifikasi nilai dari tabel validasi manual requirements.md bagian 6.

    Rumus: t_harapan = t_lancar * (1 + p_macet * 2)

    | Ruas | t_lancar | p_macet | t_harapan |
    | A-B  | 3.00     | 0.80    | 7.80      |
    | B-RS | 4.50     | 0.85    | 12.15     |
    | A-C  | 3.75     | 0.20    | 5.25      |
    | C-D  | 3.00     | 0.15    | 3.90      |
    | D-RS | 3.00     | 0.10    | 3.60      |
    """

    def test_ruas_ab(self):
        """t_harapan A-B = 3.0 * (1 + 0.80 * 2) = 3.0 * 2.6 = 7.80 menit."""
        assert abs(hitung_waktu_harapan(3.0, 0.80) - 7.80) < 1e-9

    def test_ruas_brs(self):
        """t_harapan B-RS = 4.5 * (1 + 0.85 * 2) = 4.5 * 2.7 = 12.15 menit."""
        assert abs(hitung_waktu_harapan(4.5, 0.85) - 12.15) < 1e-9

    def test_ruas_ac(self):
        """t_harapan A-C = 3.75 * (1 + 0.20 * 2) = 3.75 * 1.4 = 5.25 menit."""
        assert abs(hitung_waktu_harapan(3.75, 0.20) - 5.25) < 1e-9

    def test_ruas_cd(self):
        """t_harapan C-D = 3.0 * (1 + 0.15 * 2) = 3.0 * 1.3 = 3.90 menit."""
        assert abs(hitung_waktu_harapan(3.0, 0.15) - 3.90) < 1e-9

    def test_ruas_drs(self):
        """t_harapan D-RS = 3.0 * (1 + 0.10 * 2) = 3.0 * 1.2 = 3.60 menit."""
        assert abs(hitung_waktu_harapan(3.0, 0.10) - 3.60) < 1e-9

    def test_ruas_terbatas(self):
        """Ruas Terbatas: t_harapan dikali FAKTOR_TERBATAS (2)."""
        # t_harapan normal = 3.0 * (1 + 0.20 * 2) = 5.25
        # t_harapan terbatas = 5.25 * 2 = 10.50
        assert abs(hitung_waktu_harapan(3.75, 0.20, terbatas=True) - 10.50) < 1e-9

    def test_p_macet_nol(self):
        """p_macet=0 → t_harapan = t_lancar."""
        assert abs(hitung_waktu_harapan(3.0, 0.0) - 3.0) < 1e-9

    def test_p_macet_satu(self):
        """p_macet=1.0 → t_harapan = t_lancar * 3."""
        assert abs(hitung_waktu_harapan(3.0, 1.0) - 9.0) < 1e-9

    def test_p_macet_di_luar_rentang_error(self):
        """p_macet di luar [0,1] harus raise ValueError."""
        with pytest.raises(ValueError):
            hitung_waktu_harapan(3.0, 1.5)


# =========================================================================
# Tes hitung_biaya_ruas() dan hitung_semua_biaya()
# =========================================================================

class TestHitungBiayaRuas:
    def test_biaya_ab(self, basis):
        """Biaya A-B dengan p_macet=0.80 harus 7.80 menit."""
        assert abs(hitung_biaya_ruas(basis, "A-B", 0.80) - 7.80) < 1e-9

    def test_biaya_brs(self, basis):
        """Biaya B-RS dengan p_macet=0.85 harus 12.15 menit."""
        assert abs(hitung_biaya_ruas(basis, "B-RS", 0.85) - 12.15) < 1e-9

    def test_biaya_ac(self, basis):
        """Biaya A-C dengan p_macet=0.20 harus 5.25 menit."""
        assert abs(hitung_biaya_ruas(basis, "A-C", 0.20) - 5.25) < 1e-9

    def test_biaya_cd(self, basis):
        """Biaya C-D dengan p_macet=0.15 harus 3.90 menit."""
        assert abs(hitung_biaya_ruas(basis, "C-D", 0.15) - 3.90) < 1e-9

    def test_biaya_drs(self, basis):
        """Biaya D-RS dengan p_macet=0.10 harus 3.60 menit."""
        assert abs(hitung_biaya_ruas(basis, "D-RS", 0.10) - 3.60) < 1e-9

    def test_ruas_tidak_ada_error(self, basis):
        """Ruas yang tidak ada harus raise ValueError."""
        with pytest.raises(ValueError):
            hitung_biaya_ruas(basis, "X-Y", 0.5)


class TestHitungSemuaBiaya:
    def test_semua_biaya_manual(self, basis):
        """
        Menghitung semua biaya sekaligus dengan nilai P(macet) manual.
        Total rute A-C-D-RS harus 5.25 + 3.90 + 3.60 = 12.75 menit.
        Total rute A-B-RS harus 7.80 + 12.15 = 19.95 menit.
        """
        p_macet_manual = {
            "A-B": 0.80,
            "B-RS": 0.85,
            "A-C": 0.20,
            "C-D": 0.15,
            "D-RS": 0.10,
        }
        biaya = hitung_semua_biaya(basis, p_macet_manual, ruas_terbatas=set())

        assert abs(biaya["A-B"] - 7.80) < 1e-9
        assert abs(biaya["B-RS"] - 12.15) < 1e-9
        assert abs(biaya["A-C"] - 5.25) < 1e-9
        assert abs(biaya["C-D"] - 3.90) < 1e-9
        assert abs(biaya["D-RS"] - 3.60) < 1e-9

        total_abrs = biaya["A-B"] + biaya["B-RS"]
        total_acdrs = biaya["A-C"] + biaya["C-D"] + biaya["D-RS"]

        assert abs(total_abrs - 19.95) < 1e-9, \
            f"Total A-B-RS: harapan 19.95, dapat {total_abrs:.4f}"
        assert abs(total_acdrs - 12.75) < 1e-9, \
            f"Total A-C-D-RS: harapan 12.75, dapat {total_acdrs:.4f}"

    def test_ruas_terbatas_lebih_mahal(self, basis):
        """Ruas Terbatas harus lebih mahal dari ruas normal dengan P(macet) yang sama."""
        p = {"A-C": 0.20}
        biaya_normal = hitung_semua_biaya(basis, p, ruas_terbatas=set())
        biaya_terbatas = hitung_semua_biaya(basis, p, ruas_terbatas={"A-C"})
        assert biaya_terbatas["A-C"] > biaya_normal["A-C"]
        assert abs(biaya_terbatas["A-C"] - biaya_normal["A-C"] * 2) < 1e-9
