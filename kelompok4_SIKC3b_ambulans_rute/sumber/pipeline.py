"""
pipeline.py — Menggabungkan FOL → Bayesian Network → Biaya → A* dalam satu alur.

Alur data:
1. FOL (fol.py)       : membaca fakta dari BasisPengetahuan, menentukan status ruas
2. Graf diperbarui    : ruas Terblokir dihapus dari biaya, ruas Terbatas diberi pengali ×2
3. Bayesian Network   : menghitung P(macet) untuk ruas yang masih dapat dilalui
4. Biaya ruas         : t_harapan = t_lancar * (1 + p_macet * FAKTOR_MACET)
5. A* Search          : mencari rute tercepat, BFS, Dijkstra, terpendek jarak
6. Simpan turunan     : hasil disimpan ke BasisPengetahuan.turunan

Fungsi utama: jalankan(basis) → dict hasil lengkap
"""

from sumber.fol import jalankan_fol_dari_basis
from sumber.bayes import (
    buat_model_bayes,
    hitung_p_macet_semua_ruas,
    diagnostik_cuaca,
    CUACA_TIDAK_DIKETAHUI,
)
from sumber.biaya import hitung_semua_biaya
from sumber.graf import buat_graf
from sumber.astar import cari_astar, cari_bfs, cari_dijkstra, cari_terpendek_jarak


def jalankan(basis) -> dict:
    """
    Menjalankan pipeline lengkap AmbuRoute: FOL → Bayes → Biaya → A*.

    Membaca semua data dari BasisPengetahuan:
    - Pengetahuan tetap  : graf, CPT, aturan FOL
    - Pengetahuan dinamis: fakta lapangan, konteks, titik awal/tujuan, override manual

    Menyimpan semua hasil ke BasisPengetahuan.turunan via simpan_turunan().

    Parameter:
        basis : objek BasisPengetahuan yang sudah dimuat dan diatur konteks/faktanya

    Kembalian:
        dict {
            "rs_tidak_terjangkau"   : bool
            "pesan"                 : str (pesan ramah jika RS tidak terjangkau)
            "ruas_terblokir"        : set
            "ruas_terbatas"         : set
            "ruas_dapat_dilalui"    : set
            "jejak_inferensi"       : list of str
            "p_macet_per_ruas"      : dict {id_ruas: float}
            "biaya_per_ruas"        : dict {id_ruas: float}
            "marginalisasi_cuaca"   : dict atau None (rincian if Cuaca tidak diketahui)
            "hasil_astar"           : dict atau None
            "hasil_bfs"             : dict atau None
            "hasil_dijkstra"        : dict atau None
            "hasil_terpendek_jarak" : dict atau None
            "titik_awal"            : str
            "tujuan"                : str
        }
    """
    titik_awal = basis.dinamis.get("titik_awal") or "A"
    tujuan = basis.dinamis.get("tujuan") or "RS"

    # -------------------------------------------------------------------------
    # LANGKAH 1: FOL — tentukan status ruas
    # -------------------------------------------------------------------------
    hasil_fol = jalankan_fol_dari_basis(basis)
    ruas_terblokir = hasil_fol["ruas_terblokir"]
    ruas_terbatas = hasil_fol["ruas_terbatas"]
    ruas_dapat_dilalui = hasil_fol["ruas_dapat_dilalui"]
    rs_tidak_terjangkau = hasil_fol["rs_tidak_terjangkau"]
    jejak_inferensi = hasil_fol["jejak_inferensi"]

    # Jika RS tidak terjangkau, hentikan dan kembalikan pesan jelas
    if rs_tidak_terjangkau:
        hasil = {
            "rs_tidak_terjangkau": True,
            "pesan": (
                f"Rumah Sakit tidak dapat dijangkau dari {titik_awal}. "
                f"Semua jalur menuju {tujuan} atau dari {titik_awal} terblokir. "
                "Silakan hubungi koordinator untuk rute alternatif."
            ),
            "ruas_terblokir": ruas_terblokir,
            "ruas_terbatas": ruas_terbatas,
            "ruas_dapat_dilalui": ruas_dapat_dilalui,
            "jejak_inferensi": jejak_inferensi,
            "p_macet_per_ruas": {},
            "biaya_per_ruas": {},
            "marginalisasi_cuaca": None,
            "hasil_astar": None,
            "hasil_bfs": None,
            "hasil_dijkstra": None,
            "hasil_terpendek_jarak": None,
            "titik_awal": titik_awal,
            "tujuan": tujuan,
        }
        basis.simpan_turunan(
            ruas_terblokir=ruas_terblokir,
            ruas_terbatas=ruas_terbatas,
            rs_tidak_terjangkau=True,
            jejak_inferensi=jejak_inferensi,
            p_macet_per_ruas={},
            biaya_per_ruas={},
            hasil_astar=None,
            hasil_bfs=None,
            hasil_dijkstra=None,
            hasil_terpendek_jarak=None,
        )
        return hasil

    # -------------------------------------------------------------------------
    # LANGKAH 2: Bangun graf networkx dari pengetahuan tetap
    # -------------------------------------------------------------------------
    graf = buat_graf(basis)

    # -------------------------------------------------------------------------
    # LANGKAH 3: Bayesian Network — hitung P(macet) per ruas
    # Jika ada override manual, nilai manual dipakai untuk ruas tersebut
    # -------------------------------------------------------------------------
    model_bayes = buat_model_bayes(basis)
    p_macet_per_ruas_semua = hitung_p_macet_semua_ruas(basis, model_bayes)

    # Hanya hitung P(macet) untuk ruas yang DapatDilalui
    p_macet_per_ruas = {
        id_ruas: p
        for id_ruas, p in p_macet_per_ruas_semua.items()
        if id_ruas in ruas_dapat_dilalui
    }

    # Rincian marginalisasi + posterior — hanya saat Cuaca tidak diketahui
    konteks = basis.dinamis.get("konteks", {})
    marginalisasi_cuaca = None
    if konteks.get("cuaca") == CUACA_TIDAK_DIKETAHUI:
        jam_ctx = konteks.get("jam") or "Normal"
        pasar_ctx = konteks.get("hari_pasar") or "Tidak"
        marginalisasi_cuaca = {
            tipe: diagnostik_cuaca(model_bayes, jam_ctx, pasar_ctx, tipe)
            for tipe in ("utama", "lokal")
        }

    # -------------------------------------------------------------------------
    # LANGKAH 4: Hitung biaya ruas (waktu harapan)
    # Ruas Terbatas mendapat pengali tambahan ×2
    # -------------------------------------------------------------------------
    biaya_per_ruas = hitung_semua_biaya(basis, p_macet_per_ruas, ruas_terbatas)

    # -------------------------------------------------------------------------
    # LANGKAH 5: Cari rute dengan berbagai algoritma
    # -------------------------------------------------------------------------
    hasil_astar = cari_astar(graf, biaya_per_ruas, titik_awal, tujuan)
    hasil_bfs = cari_bfs(graf, biaya_per_ruas, titik_awal, tujuan)
    hasil_dijkstra = cari_dijkstra(graf, biaya_per_ruas, titik_awal, tujuan)
    hasil_terpendek_jarak = cari_terpendek_jarak(graf, biaya_per_ruas, titik_awal, tujuan)

    # -------------------------------------------------------------------------
    # LANGKAH 6: Simpan semua hasil ke pengetahuan turunan
    # -------------------------------------------------------------------------
    basis.simpan_turunan(
        ruas_terblokir=ruas_terblokir,
        ruas_terbatas=ruas_terbatas,
        rs_tidak_terjangkau=False,
        jejak_inferensi=jejak_inferensi,
        p_macet_per_ruas=p_macet_per_ruas,
        biaya_per_ruas=biaya_per_ruas,
        hasil_astar=hasil_astar,
        hasil_bfs=hasil_bfs,
        hasil_dijkstra=hasil_dijkstra,
        hasil_terpendek_jarak=hasil_terpendek_jarak,
    )

    return {
        "rs_tidak_terjangkau": False,
        "pesan": "",
        "ruas_terblokir": ruas_terblokir,
        "ruas_terbatas": ruas_terbatas,
        "ruas_dapat_dilalui": ruas_dapat_dilalui,
        "jejak_inferensi": jejak_inferensi,
        "p_macet_per_ruas": p_macet_per_ruas,
        "biaya_per_ruas": biaya_per_ruas,
        "marginalisasi_cuaca": marginalisasi_cuaca,
        "hasil_astar": hasil_astar,
        "hasil_bfs": hasil_bfs,
        "hasil_dijkstra": hasil_dijkstra,
        "hasil_terpendek_jarak": hasil_terpendek_jarak,
        "titik_awal": titik_awal,
        "tujuan": tujuan,
    }
