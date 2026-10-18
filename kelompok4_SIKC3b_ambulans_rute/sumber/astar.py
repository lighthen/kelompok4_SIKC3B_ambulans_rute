"""
astar.py — Implementasi A*, BFS, Dijkstra, dan rute terpendek jarak.

A* diimplementasikan sendiri menggunakan heapq agar bisa melacak:
- urutan ekspansi simpul
- jumlah simpul yang diekspansi
- waktu komputasi

networkx dipakai hanya untuk menyimpan graf dan untuk memverifikasi
hasil A* di dalam tes (bukan sebagai implementasi utama).

Heuristik A*:
    h(n) = jarak_lurus(n, RS) / kecepatan_maks * 60  (menit)

Heuristik ini admissible karena:
    t_harapan >= t_lancar = jarak / kecepatan >= jarak_lurus / kecepatan_maks
    sehingga h(n) <= biaya_aktual(n, RS) untuk semua simpul n.
"""

import heapq
import math
import time
from typing import Optional

import networkx as nx

from sumber.graf import jarak_lurus, kecepatan_maksimum, ambil_id_ruas


# =========================================================================
# TIPE KEMBALIAN
# =========================================================================

def _buat_hasil(
    rute: list,
    total_waktu_menit: float,
    total_jarak_km: float,
    jumlah_diekspansi: int,
    waktu_komputasi_detik: float,
    urutan_ekspansi: list,
    jejak: Optional[list] = None,
) -> dict:
    """
    Membungkus hasil pencarian rute dalam format dict yang seragam.

    Parameter jejak bersifat opsional dan hanya diisi oleh A* (lihat cari_astar).
    Penambahan field "jejak" tidak mengubah field lama maupun rute/biaya.
    """
    return {
        "rute": rute,
        "total_waktu_menit": total_waktu_menit,
        "total_jarak_km": total_jarak_km,
        "jumlah_diekspansi": jumlah_diekspansi,
        "waktu_komputasi_detik": waktu_komputasi_detik,
        "urutan_ekspansi": urutan_ekspansi,
        "jejak": jejak if jejak is not None else [],
    }


def _tidak_ada_rute() -> dict:
    """Hasil kosong jika rute tidak ditemukan."""
    return {
        "rute": [],
        "total_waktu_menit": float("inf"),
        "total_jarak_km": float("inf"),
        "jumlah_diekspansi": 0,
        "waktu_komputasi_detik": 0.0,
        "urutan_ekspansi": [],
        "jejak": [],
    }


# =========================================================================
# HEURISTIK
# =========================================================================

def heuristik(graf: nx.Graph, simpul: str, tujuan: str) -> float:
    """
    Menghitung nilai heuristik h(n) untuk A*.

    Rumus: h(n) = jarak_lurus(n, tujuan) / kecepatan_maks * 60  (menit)

    Heuristik ini admissible: tidak pernah melebih-lebihkan biaya sebenarnya
    karena waktu harapan >= waktu lancar >= jarak_lurus / kecepatan_maks * 60.

    Parameter:
        graf   : objek networkx.Graph
        simpul : ID simpul yang sedang dievaluasi
        tujuan : ID simpul tujuan (RS)

    Kembalian:
        estimasi waktu minimum dalam menit (float)
    """
    jarak = jarak_lurus(graf, simpul, tujuan)
    kec_maks = kecepatan_maksimum(graf)
    return (jarak / kec_maks) * 60.0


# =========================================================================
# A* SEARCH
# =========================================================================

def cari_astar(
    graf: nx.Graph,
    biaya_ruas: dict,
    titik_awal: str,
    tujuan: str,
) -> dict:
    """
    Mencari rute dengan waktu harapan minimum menggunakan algoritma A*.

    Implementasi sendiri dengan heapq, bukan networkx.astar_path,
    agar bisa melacak jumlah ekspansi dan urutan ekspansi.

    Parameter:
        graf       : networkx.Graph dengan atribut jarak_km per ruas
        biaya_ruas : dict {id_ruas: waktu_harapan_menit} dari biaya.py
        titik_awal : ID simpul awal
        tujuan     : ID simpul tujuan

    Kembalian:
        dict hasil (lihat _buat_hasil)
    """
    mulai = time.perf_counter()

    # g[simpul] = biaya terbaik dari titik_awal ke simpul
    g = {titik_awal: 0.0}
    # parent[simpul] = simpul sebelumnya di jalur terbaik
    parent = {titik_awal: None}
    # Antrian prioritas: (f, simpul)
    antrian = [(heuristik(graf, titik_awal, tujuan), titik_awal)]
    dikunjungi = set()
    urutan_ekspansi = []
    # jejak = jejak per iterasi untuk ditampilkan di antarmuka (tidak mengubah
    # rute maupun biaya). Tiap entri: iterasi, simpul, g, h, f, daftar tetangga,
    # open list terurut, dan penanda apakah simpul adalah tujuan.
    jejak = []

    while antrian:
        f_saat_ini, simpul_saat_ini = heapq.heappop(antrian)

        if simpul_saat_ini in dikunjungi:
            continue
        dikunjungi.add(simpul_saat_ini)
        urutan_ekspansi.append(simpul_saat_ini)

        catatan = {
            "iterasi": len(urutan_ekspansi),
            "simpul": simpul_saat_ini,
            "g": g[simpul_saat_ini],
            "h": heuristik(graf, simpul_saat_ini, tujuan),
            "f": f_saat_ini,
            "tetangga": [],
            "open_list": [],
            "tujuan": simpul_saat_ini == tujuan,
        }

        if simpul_saat_ini == tujuan:
            jejak.append(catatan)
            # Rekonstruksi rute
            rute = _rekonstruksi_rute(parent, tujuan)
            total_waktu = g[tujuan]
            total_jarak = _hitung_total_jarak(graf, rute)
            selesai = time.perf_counter()
            return _buat_hasil(
                rute=rute,
                total_waktu_menit=total_waktu,
                total_jarak_km=total_jarak,
                jumlah_diekspansi=len(dikunjungi),
                waktu_komputasi_detik=selesai - mulai,
                urutan_ekspansi=urutan_ekspansi,
                jejak=jejak,
            )

        for tetangga in graf.neighbors(simpul_saat_ini):
            if tetangga in dikunjungi:
                catatan["tetangga"].append({
                    "tetangga": tetangga, "ruas": None,
                    "g": None, "h": None, "f": None,
                    "status": "sudah dikunjungi",
                })
                continue
            id_ruas = ambil_id_ruas(graf, simpul_saat_ini, tetangga)
            if id_ruas not in biaya_ruas:
                # Ruas terblokir — lewati
                catatan["tetangga"].append({
                    "tetangga": tetangga, "ruas": id_ruas,
                    "g": None, "h": None, "f": None,
                    "status": "terblokir",
                })
                continue
            biaya = biaya_ruas[id_ruas]
            g_baru = g[simpul_saat_ini] + biaya
            h_tetangga = heuristik(graf, tetangga, tujuan)
            f_baru = g_baru + h_tetangga
            sudah_ada = tetangga in g
            if not sudah_ada or g_baru < g[tetangga]:
                g[tetangga] = g_baru
                parent[tetangga] = simpul_saat_ini
                heapq.heappush(antrian, (f_baru, tetangga))
                status = "update" if sudah_ada else "baru"
            else:
                status = "tidak lebih baik"
            catatan["tetangga"].append({
                "tetangga": tetangga, "ruas": id_ruas,
                "g": g_baru, "h": h_tetangga, "f": f_baru,
                "status": status,
            })

        catatan["open_list"] = sorted(
            [(f, n) for f, n in antrian], key=lambda x: (x[0], x[1])
        )
        jejak.append(catatan)

    # Tidak ditemukan rute
    selesai = time.perf_counter()
    hasil = _tidak_ada_rute()
    hasil["waktu_komputasi_detik"] = selesai - mulai
    hasil["jumlah_diekspansi"] = len(dikunjungi)
    hasil["jejak"] = jejak
    return hasil


# =========================================================================
# BFS (jumlah ruas paling sedikit)
# =========================================================================

def cari_bfs(
    graf: nx.Graph,
    biaya_ruas: dict,
    titik_awal: str,
    tujuan: str,
) -> dict:
    """
    Mencari rute dengan jumlah ruas paling sedikit menggunakan BFS.

    Ruas terblokir (tidak ada di biaya_ruas) dilewati.

    Parameter:
        graf       : networkx.Graph
        biaya_ruas : dict {id_ruas: waktu_harapan} — hanya ruas yang tersedia
        titik_awal : ID simpul awal
        tujuan     : ID simpul tujuan

    Kembalian:
        dict hasil dengan total_waktu_menit dihitung dari biaya_ruas
    """
    mulai = time.perf_counter()
    from collections import deque

    antrian = deque([(titik_awal, [titik_awal])])
    dikunjungi = {titik_awal}
    urutan_ekspansi = []

    while antrian:
        simpul, rute = antrian.popleft()
        urutan_ekspansi.append(simpul)

        if simpul == tujuan:
            total_waktu = _hitung_total_waktu(rute, biaya_ruas, graf)
            total_jarak = _hitung_total_jarak(graf, rute)
            selesai = time.perf_counter()
            return _buat_hasil(
                rute=rute,
                total_waktu_menit=total_waktu,
                total_jarak_km=total_jarak,
                jumlah_diekspansi=len(urutan_ekspansi),
                waktu_komputasi_detik=selesai - mulai,
                urutan_ekspansi=urutan_ekspansi,
            )

        for tetangga in graf.neighbors(simpul):
            if tetangga in dikunjungi:
                continue
            id_ruas = ambil_id_ruas(graf, simpul, tetangga)
            if id_ruas not in biaya_ruas:
                continue  # Ruas terblokir
            dikunjungi.add(tetangga)
            antrian.append((tetangga, rute + [tetangga]))

    selesai = time.perf_counter()
    hasil = _tidak_ada_rute()
    hasil["waktu_komputasi_detik"] = selesai - mulai
    return hasil


# =========================================================================
# DIJKSTRA (waktu harapan minimum, tanpa heuristik)
# =========================================================================

def cari_dijkstra(
    graf: nx.Graph,
    biaya_ruas: dict,
    titik_awal: str,
    tujuan: str,
) -> dict:
    """
    Mencari rute dengan waktu harapan minimum menggunakan Dijkstra.
    Sama seperti A* tetapi tanpa heuristik (h=0).

    Parameter:
        graf       : networkx.Graph
        biaya_ruas : dict {id_ruas: waktu_harapan}
        titik_awal : ID simpul awal
        tujuan     : ID simpul tujuan

    Kembalian:
        dict hasil
    """
    mulai = time.perf_counter()

    dist = {titik_awal: 0.0}
    parent = {titik_awal: None}
    antrian = [(0.0, titik_awal)]
    dikunjungi = set()
    urutan_ekspansi = []

    while antrian:
        d_saat_ini, simpul_saat_ini = heapq.heappop(antrian)

        if simpul_saat_ini in dikunjungi:
            continue
        dikunjungi.add(simpul_saat_ini)
        urutan_ekspansi.append(simpul_saat_ini)

        if simpul_saat_ini == tujuan:
            rute = _rekonstruksi_rute(parent, tujuan)
            total_jarak = _hitung_total_jarak(graf, rute)
            selesai = time.perf_counter()
            return _buat_hasil(
                rute=rute,
                total_waktu_menit=dist[tujuan],
                total_jarak_km=total_jarak,
                jumlah_diekspansi=len(dikunjungi),
                waktu_komputasi_detik=selesai - mulai,
                urutan_ekspansi=urutan_ekspansi,
            )

        for tetangga in graf.neighbors(simpul_saat_ini):
            if tetangga in dikunjungi:
                continue
            id_ruas = ambil_id_ruas(graf, simpul_saat_ini, tetangga)
            if id_ruas not in biaya_ruas:
                continue
            d_baru = dist[simpul_saat_ini] + biaya_ruas[id_ruas]
            if tetangga not in dist or d_baru < dist[tetangga]:
                dist[tetangga] = d_baru
                parent[tetangga] = simpul_saat_ini
                heapq.heappush(antrian, (d_baru, tetangga))

    selesai = time.perf_counter()
    hasil = _tidak_ada_rute()
    hasil["waktu_komputasi_detik"] = selesai - mulai
    return hasil


# =========================================================================
# RUTE TERPENDEK BERDASARKAN JARAK
# =========================================================================

def cari_terpendek_jarak(
    graf: nx.Graph,
    biaya_ruas: dict,
    titik_awal: str,
    tujuan: str,
) -> dict:
    """
    Mencari rute dengan jarak tempuh (km) terpendek menggunakan Dijkstra
    berbobot jarak, bukan waktu.

    Dipakai sebagai pembanding untuk menunjukkan bahwa rute terpendek
    berdasarkan jarak belum tentu tercepat.

    Parameter:
        graf       : networkx.Graph
        biaya_ruas : dict {id_ruas: waktu_harapan} — hanya untuk filter ruas aktif
        titik_awal : ID simpul awal
        tujuan     : ID simpul tujuan

    Kembalian:
        dict hasil dengan total_waktu_menit dihitung dari biaya_ruas
    """
    mulai = time.perf_counter()

    dist_km = {titik_awal: 0.0}
    parent = {titik_awal: None}
    antrian = [(0.0, titik_awal)]
    dikunjungi = set()
    urutan_ekspansi = []

    while antrian:
        d_saat_ini, simpul_saat_ini = heapq.heappop(antrian)

        if simpul_saat_ini in dikunjungi:
            continue
        dikunjungi.add(simpul_saat_ini)
        urutan_ekspansi.append(simpul_saat_ini)

        if simpul_saat_ini == tujuan:
            rute = _rekonstruksi_rute(parent, tujuan)
            total_waktu = _hitung_total_waktu(rute, biaya_ruas, graf)
            selesai = time.perf_counter()
            return _buat_hasil(
                rute=rute,
                total_waktu_menit=total_waktu,
                total_jarak_km=dist_km[tujuan],
                jumlah_diekspansi=len(dikunjungi),
                waktu_komputasi_detik=selesai - mulai,
                urutan_ekspansi=urutan_ekspansi,
            )

        for tetangga in graf.neighbors(simpul_saat_ini):
            if tetangga in dikunjungi:
                continue
            data = graf.get_edge_data(simpul_saat_ini, tetangga) or \
                   graf.get_edge_data(tetangga, simpul_saat_ini)
            id_ruas = data["id"]
            if id_ruas not in biaya_ruas:
                continue
            jarak = data["jarak_km"]
            d_baru = dist_km[simpul_saat_ini] + jarak
            if tetangga not in dist_km or d_baru < dist_km[tetangga]:
                dist_km[tetangga] = d_baru
                parent[tetangga] = simpul_saat_ini
                heapq.heappush(antrian, (d_baru, tetangga))

    selesai = time.perf_counter()
    hasil = _tidak_ada_rute()
    hasil["waktu_komputasi_detik"] = selesai - mulai
    return hasil


# =========================================================================
# VERIFIKASI ADMISSIBLE
# =========================================================================

def verifikasi_admissible(
    graf: nx.Graph,
    biaya_ruas: dict,
    tujuan: str,
) -> dict:
    """
    Memverifikasi bahwa heuristik A* bersifat admissible untuk semua simpul.

    Syarat admissible: h(n) <= biaya_aktual(n, tujuan) untuk semua n.
    Biaya aktual dihitung dengan Dijkstra dari setiap simpul ke tujuan.

    Parameter:
        graf       : networkx.Graph
        biaya_ruas : dict {id_ruas: waktu_harapan}
        tujuan     : ID simpul tujuan

    Kembalian:
        dict {
            "admissible": bool,
            "detail": list of dict per simpul
        }
    """
    detail = []
    admissible = True

    for simpul in graf.nodes:
        if simpul == tujuan:
            detail.append({
                "simpul": simpul,
                "h": 0.0,
                "biaya_aktual": 0.0,
                "admissible": True,
            })
            continue

        h = heuristik(graf, simpul, tujuan)
        hasil = cari_dijkstra(graf, biaya_ruas, simpul, tujuan)
        biaya_aktual = hasil["total_waktu_menit"]

        simpul_admissible = h <= biaya_aktual + 1e-9
        if not simpul_admissible:
            admissible = False

        detail.append({
            "simpul": simpul,
            "h": h,
            "biaya_aktual": biaya_aktual,
            "admissible": simpul_admissible,
        })

    return {"admissible": admissible, "detail": detail}


# =========================================================================
# FUNGSI BANTU INTERNAL
# =========================================================================

def _rekonstruksi_rute(parent: dict, tujuan: str) -> list:
    """Merekonstruksi daftar simpul rute dari dict parent."""
    rute = []
    simpul = tujuan
    while simpul is not None:
        rute.append(simpul)
        simpul = parent[simpul]
    rute.reverse()
    return rute


def _hitung_total_jarak(graf: nx.Graph, rute: list) -> float:
    """Menghitung total jarak (km) dari daftar simpul rute."""
    total = 0.0
    for i in range(len(rute) - 1):
        data = graf.get_edge_data(rute[i], rute[i + 1]) or \
               graf.get_edge_data(rute[i + 1], rute[i])
        if data:
            total += data["jarak_km"]
    return total


def _hitung_total_waktu(rute: list, biaya_ruas: dict, graf: nx.Graph) -> float:
    """Menghitung total waktu harapan (menit) dari daftar simpul rute."""
    total = 0.0
    for i in range(len(rute) - 1):
        id_ruas = ambil_id_ruas(graf, rute[i], rute[i + 1])
        total += biaya_ruas.get(id_ruas, float("inf"))
    return total
