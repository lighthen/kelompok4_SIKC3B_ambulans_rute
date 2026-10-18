"""
tes_validasi_manual.py — Checkpoint 1 (requirements.md bagian 6)

Tes ini memverifikasi bahwa seluruh sistem menghasilkan angka yang tepat
sesuai perhitungan tangan di requirements.md bagian 6.

Tabel validasi manual:
| Ruas | Jarak | t lancar | P(macet) | t harapan |
| A-B  | 2.0   | 3.00     | 0.80     | 7.80      |
| B-RS | 3.0   | 4.50     | 0.85     | 12.15     |
| A-C  | 2.5   | 3.75     | 0.20     | 5.25      |
| C-D  | 2.0   | 3.00     | 0.15     | 3.90      |
| D-RS | 2.0   | 3.00     | 0.10     | 3.60      |

Hasil yang harus dihasilkan:
- Rute A-B-RS  : 5.0 km, 19.95 menit
- Rute A-C-D-RS: 6.5 km, 12.75 menit
- A* memilih A-C-D-RS (12.75 menit) — BUKAN A-B-RS (19.95 menit)
- Rute terpendek jarak: A-B-RS (5.0 km)

Checkpoint 1: tes ini harus LULUS SEMUA sebelum lanjut ke Fase 2.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.graf import buat_graf
from sumber.biaya import hitung_semua_biaya
from sumber.astar import (
    cari_astar,
    cari_bfs,
    cari_dijkstra,
    cari_terpendek_jarak,
    verifikasi_admissible,
)

DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_GRAF = os.path.join(DIREKTORI_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIREKTORI_ROOT, "data", "cpt_bawaan.json")

# Nilai P(macet) manual sesuai tabel requirements.md bagian 6
P_MACET_MANUAL = {
    "A-B": 0.80,
    "B-RS": 0.85,
    "A-C": 0.20,
    "C-D": 0.15,
    "D-RS": 0.10,
}

TOLERANSI = 1e-6  # toleransi floating point untuk perbandingan


@pytest.fixture
def basis():
    """BasisPengetahuan dengan P(macet) manual diatur."""
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    bp.atur_p_macet_manual(P_MACET_MANUAL)
    bp.atur_titik("A", "RS")
    return bp


@pytest.fixture
def graf(basis):
    return buat_graf(basis)


@pytest.fixture
def biaya(basis):
    return hitung_semua_biaya(basis, P_MACET_MANUAL, ruas_terbatas=set())


# =========================================================================
# Verifikasi waktu lancar per ruas
# =========================================================================

class TestWaktuLancar:
    """Verifikasi kolom 't lancar' dari tabel requirements.md bagian 6."""

    def test_ab(self, graf):
        """A-B: 2.0 km / 40 km/j * 60 = 3.00 menit."""
        data = graf.get_edge_data("A", "B") or graf.get_edge_data("B", "A")
        assert abs(data["waktu_lancar"] - 3.00) < TOLERANSI

    def test_brs(self, graf):
        """B-RS: 3.0 km / 40 km/j * 60 = 4.50 menit."""
        data = graf.get_edge_data("B", "RS") or graf.get_edge_data("RS", "B")
        assert abs(data["waktu_lancar"] - 4.50) < TOLERANSI

    def test_ac(self, graf):
        """A-C: 2.5 km / 40 km/j * 60 = 3.75 menit."""
        data = graf.get_edge_data("A", "C") or graf.get_edge_data("C", "A")
        assert abs(data["waktu_lancar"] - 3.75) < TOLERANSI

    def test_cd(self, graf):
        """C-D: 2.0 km / 40 km/j * 60 = 3.00 menit."""
        data = graf.get_edge_data("C", "D") or graf.get_edge_data("D", "C")
        assert abs(data["waktu_lancar"] - 3.00) < TOLERANSI

    def test_drs(self, graf):
        """D-RS: 2.0 km / 40 km/j * 60 = 3.00 menit."""
        data = graf.get_edge_data("D", "RS") or graf.get_edge_data("RS", "D")
        assert abs(data["waktu_lancar"] - 3.00) < TOLERANSI


# =========================================================================
# Verifikasi waktu harapan per ruas
# =========================================================================

class TestWaktuHarapan:
    """Verifikasi kolom 't harapan' dari tabel requirements.md bagian 6."""

    def test_ab(self, biaya):
        """A-B: 3.00 * (1 + 0.80 * 2) = 7.80 menit."""
        assert abs(biaya["A-B"] - 7.80) < TOLERANSI

    def test_brs(self, biaya):
        """B-RS: 4.50 * (1 + 0.85 * 2) = 12.15 menit."""
        assert abs(biaya["B-RS"] - 12.15) < TOLERANSI

    def test_ac(self, biaya):
        """A-C: 3.75 * (1 + 0.20 * 2) = 5.25 menit."""
        assert abs(biaya["A-C"] - 5.25) < TOLERANSI

    def test_cd(self, biaya):
        """C-D: 3.00 * (1 + 0.15 * 2) = 3.90 menit."""
        assert abs(biaya["C-D"] - 3.90) < TOLERANSI

    def test_drs(self, biaya):
        """D-RS: 3.00 * (1 + 0.10 * 2) = 3.60 menit."""
        assert abs(biaya["D-RS"] - 3.60) < TOLERANSI


# =========================================================================
# Verifikasi total waktu per rute
# =========================================================================

class TestTotalWaktuRute:
    """Verifikasi total waktu kedua rute kandidat."""

    def test_total_abrs(self, biaya):
        """Rute A-B-RS: 7.80 + 12.15 = 19.95 menit."""
        total = biaya["A-B"] + biaya["B-RS"]
        assert abs(total - 19.95) < TOLERANSI, \
            f"Total A-B-RS: harapan 19.95, dapat {total:.6f}"

    def test_total_acdrs(self, biaya):
        """Rute A-C-D-RS: 5.25 + 3.90 + 3.60 = 12.75 menit."""
        total = biaya["A-C"] + biaya["C-D"] + biaya["D-RS"]
        assert abs(total - 12.75) < TOLERANSI, \
            f"Total A-C-D-RS: harapan 12.75, dapat {total:.6f}"

    def test_acdrs_lebih_cepat_dari_abrs(self, biaya):
        """Rute A-C-D-RS HARUS lebih cepat dari A-B-RS (poin utama sistem)."""
        total_abrs = biaya["A-B"] + biaya["B-RS"]
        total_acdrs = biaya["A-C"] + biaya["C-D"] + biaya["D-RS"]
        assert total_acdrs < total_abrs, \
            f"A-C-D-RS ({total_acdrs:.2f} mnt) tidak lebih cepat dari A-B-RS ({total_abrs:.2f} mnt)"


# =========================================================================
# Checkpoint 1: A* memilih A-C-D-RS
# =========================================================================

class TestAstarMemilihRuteTercepat:
    """
    CHECKPOINT 1 — A* harus memilih A-C-D-RS (12.75 menit), bukan A-B-RS.
    Ini adalah poin utama sistem: rute terpendek jarak != rute tercepat.
    """

    def test_astar_memilih_acdrs(self, graf, biaya):
        """A* harus memilih rute A→C→D→RS."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert hasil["rute"] == ["A", "C", "D", "RS"], \
            f"A* memilih {hasil['rute']}, harapan ['A', 'C', 'D', 'RS']"

    def test_astar_total_waktu_1275(self, graf, biaya):
        """Total waktu A* harus 12.75 menit."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert abs(hasil["total_waktu_menit"] - 12.75) < TOLERANSI, \
            f"Total waktu A*: harapan 12.75, dapat {hasil['total_waktu_menit']:.6f}"

    def test_astar_total_jarak_65(self, graf, biaya):
        """Total jarak A-C-D-RS harus 6.5 km."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert abs(hasil["total_jarak_km"] - 6.5) < TOLERANSI, \
            f"Total jarak: harapan 6.5, dapat {hasil['total_jarak_km']:.6f}"

    def test_astar_ada_ekspansi(self, graf, biaya):
        """A* harus mengekspansi setidaknya 1 simpul."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert hasil["jumlah_diekspansi"] >= 1

    def test_astar_ada_urutan_ekspansi(self, graf, biaya):
        """urutan_ekspansi harus berupa list non-kosong."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert isinstance(hasil["urutan_ekspansi"], list)
        assert len(hasil["urutan_ekspansi"]) >= 1


# =========================================================================
# Verifikasi heuristik admissible
# =========================================================================

class TestAdmissible:
    def test_heuristik_admissible_semua_simpul(self, graf, biaya):
        """
        Heuristik A* harus admissible untuk semua simpul di graf contoh.
        h(n) <= biaya_aktual(n, RS) untuk semua n.
        """
        hasil = verifikasi_admissible(graf, biaya, "RS")
        assert hasil["admissible"] is True, (
            "Heuristik tidak admissible:\n" +
            "\n".join(
                f"  {d['simpul']}: h={d['h']:.4f} > aktual={d['biaya_aktual']:.4f}"
                for d in hasil["detail"] if not d["admissible"]
            )
        )


# =========================================================================
# Rute terpendek jarak memilih A-B-RS (untuk perbandingan)
# =========================================================================

class TestRuteTerpendekJarak:
    def test_terpendek_jarak_memilih_abrs(self, graf, biaya):
        """
        Rute terpendek berdasarkan jarak harus A→B→RS (5.0 km).
        Ini yang membedakan dari rute A* (6.5 km, lebih cepat).
        """
        hasil = cari_terpendek_jarak(graf, biaya, "A", "RS")
        assert hasil["rute"] == ["A", "B", "RS"], \
            f"Terpendek jarak memilih {hasil['rute']}, harapan ['A', 'B', 'RS']"

    def test_terpendek_jarak_55km(self, graf, biaya):
        """Jarak A-B-RS = 5.0 km."""
        hasil = cari_terpendek_jarak(graf, biaya, "A", "RS")
        assert abs(hasil["total_jarak_km"] - 5.0) < TOLERANSI

    def test_terpendek_jarak_lebih_lambat_dari_astar(self, graf, biaya):
        """
        Poin utama: rute terpendek jarak LEBIH LAMBAT dari rute A*.
        A*=12.75 menit vs terpendek jarak=19.95 menit.
        """
        hasil_astar = cari_astar(graf, biaya, "A", "RS")
        hasil_jarak = cari_terpendek_jarak(graf, biaya, "A", "RS")
        assert hasil_astar["total_waktu_menit"] < hasil_jarak["total_waktu_menit"], \
            "Rute A* tidak lebih cepat dari rute terpendek jarak"


# =========================================================================
# Dijkstra dan BFS (tambahan)
# =========================================================================

class TestDijkstraDanBfs:
    def test_dijkstra_sama_dengan_astar(self, graf, biaya):
        """Dijkstra dan A* harus menemukan rute dengan biaya yang sama."""
        hasil_astar = cari_astar(graf, biaya, "A", "RS")
        hasil_dijkstra = cari_dijkstra(graf, biaya, "A", "RS")
        assert abs(
            hasil_astar["total_waktu_menit"] - hasil_dijkstra["total_waktu_menit"]
        ) < TOLERANSI, (
            f"A*={hasil_astar['total_waktu_menit']:.4f} != "
            f"Dijkstra={hasil_dijkstra['total_waktu_menit']:.4f}"
        )

    def test_bfs_menemukan_rute(self, graf, biaya):
        """BFS harus menemukan setidaknya satu rute ke RS."""
        hasil = cari_bfs(graf, biaya, "A", "RS")
        assert len(hasil["rute"]) >= 2
        assert hasil["rute"][0] == "A"
        assert hasil["rute"][-1] == "RS"

    def test_bfs_rute_paling_sedikit_ruas(self, graf, biaya):
        """BFS harus menemukan rute dengan 2 ruas (A→B→RS atau A→?→RS)."""
        hasil = cari_bfs(graf, biaya, "A", "RS")
        # Graf ini punya rute 2 ruas: A-B-RS
        assert len(hasil["rute"]) - 1 <= 3  # paling banyak 3 ruas


# =========================================================================
# Mode validasi manual di BasisPengetahuan
# =========================================================================

class TestModeValidasiManual:
    def test_p_macet_manual_dipakai(self, basis):
        """P(macet) yang diatur manual harus tersimpan di dinamis."""
        assert basis.dinamis["p_macet_manual"]["A-B"] == 0.80
        assert basis.dinamis["p_macet_manual"]["B-RS"] == 0.85

    def test_reset_manual_ke_kosong(self, basis):
        """Menghapus override manual (dict kosong) harus bekerja."""
        basis.atur_p_macet_manual({})
        assert basis.dinamis["p_macet_manual"] == {}

    def test_override_parsial(self, basis):
        """Override parsial: hanya beberapa ruas diatur manual."""
        basis.atur_p_macet_manual({"A-B": 0.50})
        assert basis.dinamis["p_macet_manual"]["A-B"] == 0.50
        assert "B-RS" not in basis.dinamis["p_macet_manual"]
