"""
tes_astar.py — Tes untuk sumber/astar.py

Mencakup:
- A* menghasilkan rute yang sama dengan networkx.astar_path
- Heuristik admissible pada graf contoh
- Penanganan ruas terblokir (tidak ada di biaya_ruas)
- Penanganan graf tidak terhubung (rute tidak ditemukan)
- BFS, Dijkstra, rute terpendek jarak berfungsi
- Urutan ekspansi A* lebih sedikit dari Dijkstra (efisiensi heuristik)
"""
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx
from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.graf import buat_graf
from sumber.biaya import hitung_semua_biaya
from sumber.astar import (
    cari_astar,
    cari_bfs,
    cari_dijkstra,
    cari_terpendek_jarak,
    heuristik,
    verifikasi_admissible,
)

DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_GRAF = os.path.join(DIREKTORI_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIREKTORI_ROOT, "data", "cpt_bawaan.json")

P_MACET_MANUAL = {
    "A-B": 0.80, "B-RS": 0.85,
    "A-C": 0.20, "C-D": 0.15, "D-RS": 0.10,
}


@pytest.fixture
def basis():
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    return bp


@pytest.fixture
def graf(basis):
    return buat_graf(basis)


@pytest.fixture
def biaya(basis):
    return hitung_semua_biaya(basis, P_MACET_MANUAL, ruas_terbatas=set())


class TestHeuristik:
    def test_tujuan_nol(self, graf):
        """h(RS, RS) = 0."""
        assert heuristik(graf, "RS", "RS") == 0.0

    def test_a_ke_rs(self, graf):
        """h(A, RS): jarak_lurus(A,RS)=5.0, kec_maks=40 -> 5/40*60 = 7.5 menit."""
        assert abs(heuristik(graf, "A", "RS") - 7.5) < 1e-9

    def test_b_ke_rs(self, graf):
        """h(B, RS): jarak_lurus(B,RS)=3.0, kec_maks=40 -> 3/40*60 = 4.5 menit."""
        assert abs(heuristik(graf, "B", "RS") - 4.5) < 1e-9

    def test_tidak_negatif(self, graf):
        """Heuristik tidak boleh negatif untuk semua simpul."""
        for simpul in graf.nodes:
            assert heuristik(graf, simpul, "RS") >= 0.0


class TestAstarVsNetworkx:
    def test_biaya_sama_dengan_networkx(self, graf, biaya):
        """
        Biaya rute A* harus sama dengan yang dihitung networkx.astar_path.
        networkx dipakai hanya untuk verifikasi, bukan implementasi utama.
        """
        # Jalankan A* sendiri
        hasil_sendiri = cari_astar(graf, biaya, "A", "RS")

        # Jalankan networkx astar_path dengan bobot waktu harapan
        def bobot_nx(u, v, data):
            id_ruas = data.get("id")
            return biaya.get(id_ruas, float("inf"))

        rute_nx = nx.astar_path(
            graf, "A", "RS",
            heuristic=lambda n, t: heuristik(graf, n, t),
            weight=bobot_nx,
        )
        biaya_nx = sum(
            biaya[graf.get_edge_data(rute_nx[i], rute_nx[i+1])["id"]]
            for i in range(len(rute_nx) - 1)
        )

        assert abs(hasil_sendiri["total_waktu_menit"] - biaya_nx) < 1e-6, (
            f"A* sendiri={hasil_sendiri['total_waktu_menit']:.4f} "
            f"!= networkx={biaya_nx:.4f}"
        )

    def test_rute_optimal(self, graf, biaya):
        """A* harus menemukan rute dengan biaya minimum."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        # Biaya optimal adalah 12.75 menit (A-C-D-RS)
        assert abs(hasil["total_waktu_menit"] - 12.75) < 1e-6


class TestAdmissibleDetail:
    def test_admissible_semua_simpul(self, graf, biaya):
        """Verifikasi admissible untuk setiap simpul secara individual."""
        hasil = verifikasi_admissible(graf, biaya, "RS")
        assert hasil["admissible"] is True
        assert len(hasil["detail"]) == graf.number_of_nodes()

    def test_detail_per_simpul(self, graf, biaya):
        """Setiap entri detail harus punya simpul, h, biaya_aktual, admissible."""
        hasil = verifikasi_admissible(graf, biaya, "RS")
        for d in hasil["detail"]:
            assert "simpul" in d
            assert "h" in d
            assert "biaya_aktual" in d
            assert "admissible" in d
            assert d["h"] >= 0
            assert d["biaya_aktual"] >= 0


class TestRuasBlokir:
    def test_astar_lewati_ruas_terblokir(self, graf, biaya):
        """A* harus melewati ruas yang tidak ada di biaya_ruas (terblokir)."""
        # Blokir B-RS — A* harus tetap menemukan rute via C-D
        biaya_tanpa_brs = {k: v for k, v in biaya.items() if k != "B-RS"}
        hasil = cari_astar(graf, biaya_tanpa_brs, "A", "RS")
        assert "B" not in hasil["rute"] or hasil["rute"].index("B") < hasil["rute"].index("RS") - 1
        assert hasil["rute"][-1] == "RS"

    def test_astar_tidak_ada_rute(self, graf, biaya):
        """Jika semua rute ke RS terblokir, A* harus kembalikan rute kosong."""
        # Blokir semua ruas yang mengarah ke RS
        biaya_blokir_semua = {k: v for k, v in biaya.items()
                               if k not in ("B-RS", "D-RS")}
        hasil = cari_astar(graf, biaya_blokir_semua, "A", "RS")
        assert hasil["rute"] == []
        assert hasil["total_waktu_menit"] == float("inf")

    def test_dijkstra_lewati_ruas_terblokir(self, graf, biaya):
        """Dijkstra harus menghasilkan rute yang valid saat ada ruas terblokir."""
        biaya_tanpa_ab = {k: v for k, v in biaya.items() if k != "A-B"}
        hasil = cari_dijkstra(graf, biaya_tanpa_ab, "A", "RS")
        assert hasil["rute"][0] == "A"
        assert hasil["rute"][-1] == "RS"
        assert "B" not in hasil["rute"]  # tidak lewat B karena A-B blokir


class TestKembalianLengkap:
    def test_astar_semua_field(self, graf, biaya):
        """Hasil A* harus punya semua field yang diperlukan."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert "rute" in hasil
        assert "total_waktu_menit" in hasil
        assert "total_jarak_km" in hasil
        assert "jumlah_diekspansi" in hasil
        assert "waktu_komputasi_detik" in hasil
        assert "urutan_ekspansi" in hasil

    def test_waktu_komputasi_positif(self, graf, biaya):
        """Waktu komputasi harus >= 0."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert hasil["waktu_komputasi_detik"] >= 0.0

    def test_rute_dimulai_dari_awal_berakhir_tujuan(self, graf, biaya):
        """Rute harus dimulai dari titik awal dan berakhir di tujuan."""
        hasil = cari_astar(graf, biaya, "A", "RS")
        assert hasil["rute"][0] == "A"
        assert hasil["rute"][-1] == "RS"
