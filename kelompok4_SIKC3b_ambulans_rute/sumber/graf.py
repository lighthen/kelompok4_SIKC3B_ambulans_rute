"""
graf.py — Memuat graf jalan dari BasisPengetahuan ke objek networkx.

Modul ini tidak menyimpan data sendiri. Semua data diambil dari
BasisPengetahuan yang dikirim sebagai parameter.
"""

import math
import networkx as nx

from sumber.basis_pengetahuan import BasisPengetahuan


def buat_graf(basis: BasisPengetahuan) -> nx.Graph:
    """
    Membuat objek networkx.Graph dari pengetahuan tetap BasisPengetahuan.

    Setiap ruas diberi atribut:
    - jarak_km       : jarak ruas dalam kilometer
    - tipe           : tipe ruas ('utama' atau 'lokal')
    - kecepatan_kmh  : kecepatan lancar dalam km/jam
    - waktu_lancar   : waktu tempuh tanpa macet dalam menit

    Setiap simpul diberi atribut:
    - nama           : nama simpul
    - x, y           : koordinat dalam km

    Parameter:
        basis : objek BasisPengetahuan yang sudah dimuat dari file

    Kembalian:
        graf networkx.Graph yang siap dipakai oleh A*, FOL, dan pipeline
    """
    graf = nx.Graph()

    # Tambahkan simpul beserta koordinat
    for simpul in basis.tetap["simpul"]:
        graf.add_node(
            simpul["id"],
            nama=simpul["nama"],
            x=simpul["koordinat"]["x"],
            y=simpul["koordinat"]["y"],
        )

    # Tambahkan ruas beserta atribut
    from sumber.biaya import hitung_waktu_lancar  # import lokal untuk hindari sirkular
    for ruas in basis.tetap["ruas"]:
        t_lancar = hitung_waktu_lancar(ruas["jarak_km"], ruas["kecepatan_kmh"])
        graf.add_edge(
            ruas["dari"],
            ruas["ke"],
            id=ruas["id"],
            jarak_km=ruas["jarak_km"],
            tipe=ruas["tipe"],
            kecepatan_kmh=ruas["kecepatan_kmh"],
            waktu_lancar=t_lancar,
        )

    return graf


def jarak_lurus(graf: nx.Graph, simpul_a: str, simpul_b: str) -> float:
    """
    Menghitung jarak Euclidean (garis lurus) antara dua simpul dalam km.
    Dipakai oleh heuristik A* dan verifikasi admissible.

    Parameter:
        graf     : objek networkx.Graph dengan atribut x, y pada simpul
        simpul_a : ID simpul pertama
        simpul_b : ID simpul kedua

    Kembalian:
        jarak dalam kilometer (float)
    """
    x1 = graf.nodes[simpul_a]["x"]
    y1 = graf.nodes[simpul_a]["y"]
    x2 = graf.nodes[simpul_b]["x"]
    y2 = graf.nodes[simpul_b]["y"]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def ambil_id_ruas(graf: nx.Graph, dari: str, ke: str) -> str:
    """
    Mengambil ID ruas dari atribut graf berdasarkan sepasang simpul.

    Parameter:
        graf : objek networkx.Graph
        dari : ID simpul asal
        ke   : ID simpul tujuan

    Kembalian:
        ID ruas (str), misal 'A-B'
    """
    data = graf.get_edge_data(dari, ke)
    if data is None:
        data = graf.get_edge_data(ke, dari)
    if data is None:
        raise ValueError(f"Ruas {dari}-{ke} tidak ditemukan di graf")
    return data["id"]


def kecepatan_maksimum(graf: nx.Graph) -> float:
    """
    Mengembalikan kecepatan tertinggi dari semua ruas di graf.
    Dipakai oleh heuristik A* agar bersifat admissible.

    Kembalian:
        kecepatan maksimum dalam km/jam (float)
    """
    kecepatan = [
        data["kecepatan_kmh"]
        for _, _, data in graf.edges(data=True)
    ]
    return max(kecepatan) if kecepatan else 40.0
