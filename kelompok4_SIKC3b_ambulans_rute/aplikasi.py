"""
aplikasi.py — Antarmuka Streamlit AmbuRoute

Jalankan dengan: streamlit run aplikasi.py
"""

import os
import sys
import warnings
# Sembunyikan FutureWarning dari pgmpy (bukan dari kode kita)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# Pastikan folder sumber bisa diimpor
sys.path.insert(0, os.path.dirname(__file__))

from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.pipeline import jalankan
from sumber.graf import buat_graf, jarak_lurus, kecepatan_maksimum
from sumber.astar import heuristik
from sumber.bayes import (
    prior_node_akar,
    prior_cuaca,
    mode_aktif,
    CUACA_TIDAK_DIKETAHUI,
)
from sumber.fol import (
    KECELAKAAN,
    BANJIR,
    TERTUTUP,
    LAJUR_LEBIH_DARI_SATU,
    TERHUBUNG,
    TERBLOKIR,
    TERBATAS,
    DAPAT_DILALUI,
    TERISOLASI,
    RS_TIDAK_TERJANGKAU,
)

# =========================================================================
# KONFIGURASI HALAMAN
# =========================================================================

st.set_page_config(
    page_title="AmbuRoute — Rute Tercepat Ambulans",
    page_icon="🚑",
    layout="wide",
)

# =========================================================================
# PATH FILE DATA
# =========================================================================

DIR_ROOT = os.path.dirname(__file__)
PATH_GRAF = os.path.join(DIR_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIR_ROOT, "data", "cpt_bawaan.json")
PATH_EKSPOR = os.path.join(DIR_ROOT, "dokumen", "basis_pengetahuan.md")

# =========================================================================
# INISIALISASI SESSION STATE
# =========================================================================

if "basis" not in st.session_state:
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    st.session_state.basis = bp

if "hasil" not in st.session_state:
    st.session_state.hasil = None

if "laporan_kejadian" not in st.session_state:
    st.session_state.laporan_kejadian = []  # list of dict

basis: BasisPengetahuan = st.session_state.basis

# =========================================================================
# KONSTANTA TAMPILAN
# =========================================================================

# Peta status langkah ekspansi A* ke label berbahasa Indonesia
STATUS_TRACE = {
    "baru": "🆕 baru (di-push)",
    "update": "🔄 update (di-push)",
    "tidak lebih baik": "➖ tidak lebih baik (tidak di-push)",
    "sudah dikunjungi": "⏭ sudah dikunjungi",
    "terblokir": "🚫 terblokir (dilewati)",
}

# Daftar predikat FOL, dibedakan fakta lapangan vs predikat turunan
FAKTA_LAPANGAN = [
    (KECELAKAAN, "terjadi kecelakaan di ruas r"),
    (BANJIR, "terjadi banjir di ruas r"),
    (TERTUTUP, "ruas r tertutup total"),
    (LAJUR_LEBIH_DARI_SATU, "ruas r masih punya lebih dari satu lajur"),
    (TERHUBUNG, "simpul s adalah ujung ruas r (fakta struktural dari graf)"),
]

PREDIKAT_TURUNAN = [
    (TERBLOKIR, "ruas r tidak dapat dilalui (dihapus dari graf)"),
    (TERBATAS, "ruas r dapat dilalui tetapi waktu tempuh dikali ×2"),
    (DAPAT_DILALUI, "ruas r normal (asumsi dunia tertutup)"),
    (TERISOLASI, "semua ruas yang menyentuh simpul s terblokir"),
    (RS_TIDAK_TERJANGKAU, "RS atau titik awal terisolasi"),
]

# =========================================================================
# FUNGSI BANTU
# =========================================================================

def daftar_simpul():
    return [s["id"] for s in basis.tetap["simpul"]]

def daftar_ruas():
    return [r["id"] for r in basis.tetap["ruas"]]

def gambar_graf(hasil=None):
    """
    Menggambar graf dengan matplotlib. Rute terpilih disorot biru,
    ruas terblokir merah, ruas terbatas kuning.
    """
    graf_nx = buat_graf(basis)

    # Tentukan posisi simpul dari koordinat di basis pengetahuan
    pos = {
        s["id"]: (s["koordinat"]["x"], s["koordinat"]["y"])
        for s in basis.tetap["simpul"]
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    ruas_terblokir = set()
    ruas_terbatas = set()
    rute_terpilih = []

    if hasil:
        ruas_terblokir = hasil.get("ruas_terblokir", set())
        ruas_terbatas = hasil.get("ruas_terbatas", set())
        if hasil.get("hasil_astar") and hasil["hasil_astar"]["rute"]:
            rute_terpilih = hasil["hasil_astar"]["rute"]

    # Kelompokkan ruas berdasarkan status
    ruas_normal_edges = []
    ruas_terbatas_edges = []
    ruas_terblokir_edges = []
    ruas_rute_edges = []

    for u, v, data in graf_nx.edges(data=True):
        id_r = data["id"]
        pasang = (u, v)
        if id_r in ruas_terblokir:
            ruas_terblokir_edges.append(pasang)
        elif id_r in ruas_terbatas:
            ruas_terbatas_edges.append(pasang)
        else:
            ruas_normal_edges.append(pasang)

    # Tandai ruas rute terpilih: keluarkan dari kelompok status mana pun agar
    # tidak digambar dua kali (termasuk ruas Terbatas yang dipakai rute).
    if len(rute_terpilih) >= 2:
        for i in range(len(rute_terpilih) - 1):
            pasang = (rute_terpilih[i], rute_terpilih[i+1])
            pasang_balik = (rute_terpilih[i+1], rute_terpilih[i])
            for grup in (ruas_normal_edges, ruas_terbatas_edges, ruas_terblokir_edges):
                if pasang in grup:
                    grup.remove(pasang)
                elif pasang_balik in grup:
                    grup.remove(pasang_balik)
            ruas_rute_edges.append(pasang)

    # Gambar ruas
    nx.draw_networkx_edges(graf_nx, pos, edgelist=ruas_normal_edges,
                           edge_color="#adb5bd", width=2, ax=ax)
    nx.draw_networkx_edges(graf_nx, pos, edgelist=ruas_terbatas_edges,
                           edge_color="#ffc107", width=3, style="dashed", ax=ax)
    nx.draw_networkx_edges(graf_nx, pos, edgelist=ruas_terblokir_edges,
                           edge_color="#dc3545", width=3, style="dotted", ax=ax)
    nx.draw_networkx_edges(graf_nx, pos, edgelist=ruas_rute_edges,
                           edge_color="#0d6efd", width=5, ax=ax,
                           arrows=True, arrowstyle="->", arrowsize=20,
                           connectionstyle="arc3,rad=0.05")

    # Warna simpul
    warna_simpul = []
    for simpul in graf_nx.nodes:
        if simpul == "A":
            warna_simpul.append("#198754")   # hijau — titik awal
        elif simpul == "RS":
            warna_simpul.append("#dc3545")   # merah — RS
        elif hasil and simpul in hasil.get("simpul_terisolasi", set()):
            warna_simpul.append("#6c757d")   # abu — terisolasi
        elif rute_terpilih and simpul in rute_terpilih:
            warna_simpul.append("#0d6efd")   # biru — di rute
        else:
            warna_simpul.append("#fd7e14")   # oranye — simpul biasa

    nx.draw_networkx_nodes(graf_nx, pos, node_color=warna_simpul,
                           node_size=800, ax=ax)
    nx.draw_networkx_labels(graf_nx, pos, font_color="white",
                            font_weight="bold", font_size=11, ax=ax)

    # Label jarak pada ruas
    edge_labels = {
        (ruas_data["dari"], ruas_data["ke"]): f"{ruas_data['jarak_km']} km"
        for ruas_data in basis.tetap["ruas"]
    }
    nx.draw_networkx_edge_labels(graf_nx, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax, label_pos=0.3)

    # Legenda
    legenda = [
        mpatches.Patch(color="#0d6efd", label="Rute terpilih (A*)"),
        mpatches.Patch(color="#adb5bd", label="Ruas normal"),
        mpatches.Patch(color="#ffc107", label="Ruas terbatas (×2)"),
        mpatches.Patch(color="#dc3545", label="Ruas terblokir"),
        mpatches.Patch(color="#198754", label="Titik awal (A)"),
        mpatches.Patch(color="#dc3545", label="Rumah Sakit (RS)"),
    ]
    # Legenda ditaruh di pojok kiri bawah agar tidak menutupi simpul A (0,0)
    ax.legend(handles=legenda, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Graf Jalan AmbuRoute", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def gambar_dag_bayes():
    """
    Menggambar diagram DAG Bayesian Network:
    Jam, Cuaca, HariPasar, TipeRuas → Macet.
    """
    dag = nx.DiGraph()
    akar = ["Jam", "Cuaca", "HariPasar", "TipeRuas"]
    for a in akar:
        dag.add_edge(a, "Macet")

    pos = {
        "Jam": (0, 1),
        "Cuaca": (1, 1),
        "HariPasar": (2, 1),
        "TipeRuas": (3, 1),
        "Macet": (1.5, 0),
    }

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#f8f9fa")
    nx.draw_networkx_nodes(dag, pos, node_color="#0d6efd",
                           node_size=1600, ax=ax)
    nx.draw_networkx_nodes(dag, pos, nodelist=["Macet"],
                           node_color="#dc3545", node_size=1600, ax=ax)
    nx.draw_networkx_edges(dag, pos, arrows=True, arrowstyle="-|>",
                           arrowsize=18, edge_color="#495057", width=2, ax=ax)
    nx.draw_networkx_labels(dag, pos, font_color="white",
                            font_weight="bold", font_size=9, ax=ax)
    ax.set_title("DAG Bayesian Network: 4 induk → Macet",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def status_label(id_ruas, hasil):
    """Mengembalikan label status ruas untuk tabel."""
    if not hasil:
        return "—"
    if id_ruas in hasil.get("ruas_terblokir", set()):
        return "🔴 Terblokir"
    if id_ruas in hasil.get("ruas_terbatas", set()):
        return "🟡 Terbatas (×2)"
    if id_ruas in hasil.get("ruas_dapat_dilalui", set()):
        return "🟢 Dapat Dilalui"
    return "—"


def tabel_h_dan_konsistensi(graf_nx, tujuan):
    """
    Membuat tabel nilai h(n) dan tabel cek konsistensi
    h(n) <= c(n,n') + h(n') untuk tiap ruas dua arah, dengan c = t_lancar
    (batas bawah biaya).
    """
    kec = kecepatan_maksimum(graf_nx)

    baris_h = []
    for n in graf_nx.nodes:
        d = jarak_lurus(graf_nx, n, tujuan)
        baris_h.append({
            "Simpul": n,
            "Jarak Lurus ke tujuan (km)": f"{d:.3f}",
            "h(n) (menit)": f"{d / kec * 60:.4f}",
        })

    baris_konsistensi = []
    for u, v, data in graf_nx.edges(data=True):
        t_lancar = data["waktu_lancar"]
        for a, b in ((u, v), (v, u)):
            ha = heuristik(graf_nx, a, tujuan)
            hb = heuristik(graf_nx, b, tujuan)
            selisih = ha - hb
            konsisten = selisih <= t_lancar + 1e-9
            baris_konsistensi.append({
                "Ruas": data["id"],
                "Arah": f"{a} → {b}",
                "h(n)": f"{ha:.4f}",
                "h(n')": f"{hb:.4f}",
                "h(n) − h(n')": f"{selisih:.4f}",
                "t_lancar = c(n,n') (menit)": f"{t_lancar:.4f}",
                "Konsisten": "✅" if konsisten else "❌",
            })

    return baris_h, baris_konsistensi


def baris_perbandingan(nama, hasil_algo):
    """Membentuk satu baris tabel perbandingan algoritma."""
    if not hasil_algo or not hasil_algo.get("rute"):
        return {
            "Algoritma": nama,
            "Rute": "Tidak ditemukan",
            "Waktu (menit)": "—",
            "Jarak (km)": "—",
            "Simpul Diekspansi": "—",
            "Komputasi (ms)": "—",
        }
    return {
        "Algoritma": nama,
        "Rute": " → ".join(hasil_algo["rute"]),
        "Waktu (menit)": f"{hasil_algo['total_waktu_menit']:.2f}",
        "Jarak (km)": f"{hasil_algo['total_jarak_km']:.1f}",
        "Simpul Diekspansi": hasil_algo["jumlah_diekspansi"],
        "Komputasi (ms)": f"{hasil_algo['waktu_komputasi_detik'] * 1000:.3f}",
    }


# =========================================================================
# HEADER
# =========================================================================

st.title("AmbuRoute")
st.caption("Kelompok 4 — SIKC3B")
st.caption(
    "Sistem pencarian rute terbaik untuk pertolongan pertama pasien "
    "menggunakan FOL, Bayesian Network, dan A\\* Search"
)
st.divider()

# =========================================================================
# SIDEBAR — INPUT
# =========================================================================

with st.sidebar:
    st.header("Pengaturan Kondisi")

    col1, col2 = st.columns(2)
    with col1:
        jam = st.selectbox("Jam", ["Sibuk", "Normal"], key="jam")
        cuaca = st.selectbox(
            "Cuaca", ["Hujan", "Cerah", CUACA_TIDAK_DIKETAHUI], key="cuaca"
        )
    with col2:
        hari_pasar = st.selectbox("Hari Pasar", ["Ya", "Tidak"], key="hari_pasar")

    if cuaca == CUACA_TIDAK_DIKETAHUI:
        st.caption(
            "Cuaca tidak diketahui → P(Macet) dihitung dengan **marginalisasi** "
            "(Hukum Probabilitas Total) memakai prior Cuaca **ASUMSI** "
            f"(Hujan={prior_cuaca()['Hujan']:.2f}, Cerah={prior_cuaca()['Cerah']:.2f})."
        )

    st.divider()
    st.subheader("Titik Perjalanan")
    simpul_list = daftar_simpul()
    titik_awal = st.selectbox("Titik Awal", simpul_list, key="titik_awal")
    tujuan = st.selectbox(
        "RS Tujuan",
        simpul_list,
        index=simpul_list.index("RS") if "RS" in simpul_list else len(simpul_list) - 1,
        key="tujuan",
    )

    st.divider()
    st.subheader("Laporkan Kejadian")

    ruas_list = daftar_ruas()
    ruas_pilih = st.selectbox("Ruas", ruas_list, key="ruas_pilih")
    jenis_pilih = st.selectbox("Jenis Kejadian", ["Kecelakaan", "Banjir"], key="jenis_pilih")

    if jenis_pilih == "Kecelakaan":
        tertutup_pilih = st.checkbox("Jalan tertutup total?", key="tertutup_pilih")
        lajur_pilih = st.checkbox("Lajur tersisa lebih dari satu?", key="lajur_pilih")
    else:
        tertutup_pilih = False
        lajur_pilih = False

    col_tambah, col_hapus = st.columns(2)
    with col_tambah:
        if st.button("+ Tambah", use_container_width=True):
            laporan_baru = {
                "ruas": ruas_pilih,
                "jenis": jenis_pilih,
                "tertutup": tertutup_pilih,
                "lajur_lebih": lajur_pilih,
            }
            # Cegah duplikat
            sudah_ada = any(
                l["ruas"] == ruas_pilih and l["jenis"] == jenis_pilih
                for l in st.session_state.laporan_kejadian
            )
            if not sudah_ada:
                st.session_state.laporan_kejadian.append(laporan_baru)

    with col_hapus:
        if st.button("- Hapus Semua", use_container_width=True):
            st.session_state.laporan_kejadian = []

    # Tampilkan daftar laporan aktif
    if st.session_state.laporan_kejadian:
        st.caption("Laporan aktif:")
        for i, lap in enumerate(st.session_state.laporan_kejadian):
            keterangan = f"{lap['jenis']} di {lap['ruas']}"
            if lap["jenis"] == "Kecelakaan":
                if lap["tertutup"]:
                    keterangan += " (tertutup)"
                elif lap["lajur_lebih"]:
                    keterangan += " (≥2 lajur)"
                else:
                    keterangan += " (1 lajur)"
            st.caption(f"  {i+1}. {keterangan}")

    st.divider()

    if st.button("Cari Rute", type="primary", use_container_width=True):
        # Validasi input
        if titik_awal == tujuan:
            st.warning("Titik awal dan tujuan tidak boleh sama.")
        else:
            try:
                # Reset dan atur basis pengetahuan
                basis.reset_dinamis()
                basis.atur_titik(titik_awal, tujuan)
                basis.atur_konteks(jam, cuaca, hari_pasar)

                # Tambahkan fakta dari laporan
                for lap in st.session_state.laporan_kejadian:
                    ruas_id = lap["ruas"]
                    if lap["jenis"] == "Banjir":
                        basis.tambah_fakta("Banjir", ruas_id)
                    else:  # Kecelakaan
                        basis.tambah_fakta("Kecelakaan", ruas_id)
                        if lap["tertutup"]:
                            basis.tambah_fakta("Tertutup", ruas_id)
                        if lap["lajur_lebih"]:
                            basis.tambah_fakta("LajurTersisaLebihDariSatu", ruas_id)

                # Jalankan pipeline
                with st.spinner("Mencari rute..."):
                    hasil = jalankan(basis)

                # Ekspor basis pengetahuan
                try:
                    basis.ekspor_markdown(PATH_EKSPOR)
                except Exception:
                    pass

                st.session_state.hasil = hasil

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

# =========================================================================
# PANEL UTAMA
# =========================================================================

hasil = st.session_state.hasil

# --- Pesan RS tidak terjangkau ---
if hasil and hasil.get("rs_tidak_terjangkau"):
    st.error(
        f"**Rumah Sakit Tidak Dapat Dijangkau**\n\n{hasil['pesan']}"
    )
    # Tetap tampilkan graf dan jejak FOL
    col_graf, col_info = st.columns([3, 2])
    with col_graf:
        fig_rs = gambar_graf(hasil)
        st.pyplot(fig_rs)
        plt.close(fig_rs)
    with col_info:
        st.subheader("Jejak Inferensi FOL")
        for langkah in hasil.get("jejak_inferensi", []):
            if langkah.startswith("["):
                st.caption(langkah)
            elif "aktif" in langkah and "tidak aktif" not in langkah:
                st.markdown(f"🔴 {langkah}")
            else:
                st.caption(langkah)
    st.stop()

# --- Tampilan normal ---
if hasil and not hasil.get("rs_tidak_terjangkau"):
    astar = hasil.get("hasil_astar", {})
    rute = astar.get("rute", [])

    # Ringkasan hasil
    if rute:
        st.success(
            f"**Rute Ditemukan:** {' → '.join(rute)}  |  "
            f"⏱**{astar['total_waktu_menit']:.2f} menit**  |  "
            f"**{astar['total_jarak_km']:.1f} km**"
        )

    # Kolom graf + ringkasan
    col_graf, col_ringkasan = st.columns([3, 2])

    with col_graf:
        st.subheader("Peta Rute")
        fig_peta = gambar_graf(hasil)
        st.pyplot(fig_peta)
        plt.close(fig_peta)

    with col_ringkasan:
        st.subheader("Ringkasan")
        if rute:
            st.metric("Estimasi Waktu (A*)", f"{astar['total_waktu_menit']:.2f} menit")
            st.metric("Total Jarak", f"{astar['total_jarak_km']:.1f} km")
            st.metric("Simpul Diekspansi", astar["jumlah_diekspansi"])

        st.caption(f"Kondisi: **{jam}** | **{cuaca}** | Pasar: **{hari_pasar}**")
        st.caption(f"Ruas terblokir: {', '.join(sorted(hasil['ruas_terblokir'])) or 'Tidak ada'}")
        st.caption(f"Ruas terbatas: {', '.join(sorted(hasil['ruas_terbatas'])) or 'Tidak ada'}")

    st.divider()

    # =====================================================================
    # TAB: Tabel Ruas | Trace A* | Jejak FOL | Basis Pengetahuan | Perbandingan
    # =====================================================================

    tab_ruas, tab_trace, tab_fol, tab_bp, tab_perbandingan = st.tabs([
        "Tabel Ruas",
        "Trace A*",
        "Jejak Inferensi FOL",
        "Basis Pengetahuan",
        "Perbandingan Algoritma",
    ])

    # --- Tab Tabel Ruas ---
    with tab_ruas:
        st.subheader("Status Ruas")
        st.caption(
            "t_lancar = jarak ÷ kecepatan × 60  |  "
            "Faktor = 1 + 2·P(Macet)  |  "
            "t_harapan = t_lancar × Faktor  |  "
            "ruas Terbatas dikali ×2 tambahan."
        )
        baris_tabel = []
        for ruas_data in basis.tetap["ruas"]:
            id_r = ruas_data["id"]
            p = hasil["p_macet_per_ruas"].get(id_r, None)
            biaya = hasil["biaya_per_ruas"].get(id_r, None)
            t_lancar = ruas_data["jarak_km"] / ruas_data["kecepatan_kmh"] * 60.0
            terbatas = id_r in hasil.get("ruas_terbatas", set())

            faktor = f"{1 + 2 * p:.2f}" if p is not None else "—"
            if terbatas:
                faktor = f"{faktor} ×2" if faktor != "—" else "×2"

            baris_tabel.append({
                "Ruas": id_r,
                "Jarak (km)": ruas_data["jarak_km"],
                "Tipe": ruas_data["tipe"],
                "Status FOL": status_label(id_r, hasil),
                "Waktu Lancar (mnt)": f"{t_lancar:.2f}",
                "P(Macet)": f"{p:.3f}" if p is not None else "—",
                "Faktor (1+2p)": faktor,
                "Waktu Harapan (mnt)": f"{biaya:.2f}" if biaya is not None else "—",
            })
        st.dataframe(baris_tabel, use_container_width=True, hide_index=True)
        st.caption(
            "P(Macet) berasal dari Bayesian Network (ASUMSI), marginalisasi "
            "Cuaca (ASUMSI), atau input validasi manual."
        )

        if hasil.get("marginalisasi_cuaca"):
            st.divider()
            st.subheader("Marginalisasi Cuaca & Posterior Diagnostik")
            st.caption(
                "Karena Cuaca **tidak diketahui**, P(Macet) dihitung dengan "
                "Hukum Probabilitas Total (marginalisasi) atas prior Cuaca "
                "**ASUMSI**. Cuaca adalah node akar yang independen pada DAG, "
                "sehingga marginalisasi ini sah."
            )
            st.latex(
                r"P(\text{Macet}\mid jam, pasar, tipe) = "
                r"\sum_{c \in \{Hujan, Cerah\}} "
                r"P(\text{Macet}\mid c, jam, pasar, tipe)\,P(c)"
            )
            st.latex(
                r"P(c \mid \text{Macet}) = "
                r"\frac{P(\text{Macet}\mid c)\;P(c)}"
                r"{\sum_{c'} P(\text{Macet}\mid c')\;P(c')}"
            )

            st.markdown("**Prior Cuaca (ASUMSI):**")
            baris_prior_cuaca = [
                {"Cuaca": c, "Prior P(c) (ASUMSI)": f"{p:.2f}"}
                for c, p in prior_cuaca().items()
            ]
            st.dataframe(baris_prior_cuaca, use_container_width=True, hide_index=True)

            for tipe, info in hasil["marginalisasi_cuaca"].items():
                st.markdown(f"**Tipe ruas: {tipe}**")
                baris_marg = []
                for b in info["baris"]:
                    baris_marg.append({
                        "Cuaca": b["cuaca"],
                        "Prior P(c) (ASUMSI)": f"{b['prior']:.2f}",
                        "P(Macet=Ya | c)": f"{b['p_macet_ya']:.4f}",
                        "Kontribusi P(Macet|c)·P(c)": f"{b['kontribusi']:.4f}",
                    })
                st.dataframe(baris_marg, use_container_width=True, hide_index=True)
                st.caption(
                    f"P(Macet) = {info['p_macet']:.4f}  |  "
                    f"posterior P(Hujan | Macet) = {info['posterior']['Hujan']:.4f}"
                )

            with st.expander("Contoh perhitungan (angka dari program)"):
                info_u = hasil["marginalisasi_cuaca"]["utama"]
                suku = " + ".join(
                    f"{b['p_macet_ya']:.4f} × {b['prior']:.2f}"
                    for b in info_u["baris"]
                )
                st.markdown(f"- P(Macet) = {suku} = {info_u['p_macet']:.4f}")
                b_hujan = info_u["baris"][0]
                st.markdown(
                    "- P(Hujan | Macet) = "
                    "P(Macet|Hujan)·P(Hujan) ÷ P(Macet) = "
                    f"{b_hujan['kontribusi']:.4f} ÷ {info_u['p_macet']:.4f} = "
                    f"{info_u['posterior']['Hujan']:.4f}"
                )

    # --- Tab Trace A* ---
    with tab_trace:
        st.subheader("Trace A* per Iterasi")
        jejak = astar.get("jejak", [])
        if not jejak:
            st.info("Tidak ada jejak iterasi (kemungkinan rute tidak ditemukan).")
        else:
            for it in jejak:
                judul = (
                    f"Iterasi {it['iterasi']}: ekspansi **{it['simpul']}** "
                    f"(g={it['g']:.2f}, h={it['h']:.2f}, f={it['f']:.2f})"
                )
                if it.get("tujuan"):
                    judul += " — ✅ TUJUAN"
                with st.expander(judul, expanded=True):
                    if it["tetangga"]:
                        baris_tetangga = []
                        for t in it["tetangga"]:
                            baris_tetangga.append({
                                "Tetangga": t["tetangga"],
                                "Ruas": t["ruas"] if t["ruas"] else "—",
                                "g'": f"{t['g']:.2f}" if t["g"] is not None else "—",
                                "h'": f"{t['h']:.2f}" if t["h"] is not None else "—",
                                "f'": f"{t['f']:.2f}" if t["f"] is not None else "—",
                                "Status": STATUS_TRACE.get(t["status"], t["status"]),
                            })
                        st.dataframe(baris_tetangga, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Tidak ada tetangga yang diperiksa (simpul tujuan).")

                    if it["open_list"]:
                        isi_open = ", ".join(f"({f:.2f}, {n})" for f, n in it["open_list"])
                    else:
                        isi_open = "(kosong)"
                    st.caption(f"Open list terurut (f, simpul): {isi_open}")

        st.divider()
        graf_nx = buat_graf(basis)
        tujuan_eval = hasil.get("tujuan", "RS")
        baris_h, baris_konsistensi = tabel_h_dan_konsistensi(graf_nx, tujuan_eval)

        st.subheader("Nilai Heuristik h(n)")
        st.caption(
            f"h(n) = jarak_lurus(n, {tujuan_eval}) ÷ kecepatan_maks × 60 (menit). "
            "kecepatan_maks = 40 km/j (dari data graf)."
        )
        st.dataframe(baris_h, use_container_width=True, hide_index=True)

        st.subheader("Cek Konsistensi h(n) ≤ c(n,n') + h(n')")
        st.caption(
            "Diuji untuk setiap ruas dua arah. c(n,n') memakai t_lancar "
            "(waktu lancar) sebagai batas bawah biaya. Konsisten ⇒ admissible."
        )
        st.dataframe(baris_konsistensi, use_container_width=True, hide_index=True)

    # --- Tab Jejak Inferensi FOL ---
    with tab_fol:
        st.subheader("Jejak Inferensi First Order Logic")
        st.caption(
            "Tiap langkah menampilkan pengikatan variabel r (ruas) atau s (simpul) "
            "ke objeknya, mis. [pengikatan: r = B-RS]."
        )
        jejak_fol = hasil.get("jejak_inferensi", [])
        if not jejak_fol:
            st.info("Tidak ada jejak inferensi.")
        else:
            # Fakta aktif
            fakta_aktif = sorted(basis.dinamis["fakta"])
            if fakta_aktif:
                st.markdown("**Fakta Lapangan Aktif:**")
                for p, o in fakta_aktif:
                    st.markdown(f"- `{p}({o})`")
                st.divider()

            st.markdown("**Langkah Inferensi:**")
            for i, langkah in enumerate(jejak_fol, 1):
                if langkah.startswith("["):
                    st.info(langkah)
                elif "tidak aktif" in langkah.lower():
                    st.caption(f"{i}. {langkah}")
                elif ("R7" in langkah) or ("Terisolasi" in langkah):
                    st.error(f"{i}. {langkah}")
                elif "aktif" in langkah:
                    st.warning(f"{i}. {langkah}")
                else:
                    st.caption(f"{i}. {langkah}")

    # --- Tab Basis Pengetahuan ---
    with tab_bp:
        st.subheader("Basis Pengetahuan Sistem")
        st.caption(f"Mode Bayesian Network aktif: **{mode_aktif()}**")

        st.markdown("#### 1. Diagram DAG Bayesian Network")
        st.caption("Empat node akar (Jam, Cuaca, HariPasar, TipeRuas) → Macet.")
        fig_dag = gambar_dag_bayes()
        st.pyplot(fig_dag)
        plt.close(fig_dag)

        st.markdown("#### 2. Conditional Probability Table (CPT)")
        st.caption(
            "16 baris CPT. **ASUMSI**: semua nilai probabilitas di bawah adalah "
            "asumsi, bukan data lapangan."
        )
        baris_cpt = []
        for e in basis.tetap["cpt"]:
            baris_cpt.append({
                "Jam": e["jam"],
                "Cuaca": e["cuaca"],
                "Hari Pasar": e["hari_pasar"],
                "Tipe Ruas": e["tipe_ruas"],
                "P(Macet=Ya)": f"{e['p_macet_ya']:.2f}",
            })
        st.dataframe(baris_cpt, use_container_width=True, hide_index=True)
        st.caption("Seluruh nilai P(Macet=Ya) diberi label ASUMSI.")

        st.markdown("#### 3. Prior Node Akar")
        prior = prior_node_akar()
        if prior:
            st.caption(
                "**ASUMSI**: prior seragam (bukan dari data). Tidak memengaruhi "
                "hasil karena keempat induk selalu dijadikan evidence."
            )
            baris_prior = []
            for var, nilai in prior.items():
                for state, p in nilai.items():
                    baris_prior.append({
                        "Variabel": var,
                        "Nilai": state,
                        "Prior": f"{p:.2f}",
                        "Label": "ASUMSI",
                    })
            st.dataframe(baris_prior, use_container_width=True, hide_index=True)
        else:
            st.info("Prior node akar hanya tersedia saat mode pgmpy (mode fallback tidak menyimpan prior).")

        st.markdown("#### 4. Predikat FOL")
        col_fakta, col_turunan = st.columns(2)
        with col_fakta:
            st.markdown("**Fakta lapangan (input pengguna):**")
            for nama, ket in FAKTA_LAPANGAN:
                st.markdown(f"- `{nama}(r)` — {ket}")
        with col_turunan:
            st.markdown("**Predikat turunan (hasil inferensi):**")
            for nama, ket in PREDIKAT_TURUNAN:
                st.markdown(f"- `{nama}` — {ket}")

        st.markdown("#### 5. Aturan FOL (R1–R7)")
        for a in basis.tetap["aturan_fol"]:
            st.markdown(f"- **{a['kode']}**: `{a['notasi']}` — {a['keterangan']}")

        with st.expander("Mengapa FOL, bukan Logika Proposisional?"):
            st.markdown(
                "- **Logika proposisional** tidak punya variabel dan kuantor, jadi "
                "satu aturan hanya berlaku untuk satu ruas tertentu. Untuk 5 ruas, "
                "R3 perlu ditulis 5 kali: `K(A-B) ∧ ¬T(A-B) ∧ ¬L(A-B) → X(A-B)`, "
                "`K(B-RS) ∧ ¬T(B-RS) ∧ ¬L(B-RS) → X(B-RS)`, dan seterusnya."
            )
            st.markdown(
                "- **First Order Logic** cukup satu aturan dengan kuantor universal "
                "`∀r` yang berlaku untuk semua ruas: "
                "`∀r Kecelakaan(r) ∧ ¬Tertutup(r) ∧ ¬LajurTersisaLebihDariSatu(r) "
                "→ Terblokir(r)`."
            )
            st.markdown(
                "- Jadi menambah ruas baru tidak perlu menambah aturan. FOL termasuk "
                "**penalaran deduktif**: dari fakta + aturan umum, kesimpulan "
                "diturunkan secara pasti (modus ponens), bukan menebak."
            )

    # --- Tab Perbandingan Algoritma ---
    with tab_perbandingan:
        st.subheader("Perbandingan Algoritma Pencarian Rute")

        tabel_perbandingan = [
            baris_perbandingan("⭐ A* (terpilih)", hasil.get("hasil_astar")),
            baris_perbandingan("Dijkstra", hasil.get("hasil_dijkstra")),
            baris_perbandingan("BFS (ruas minimum)", hasil.get("hasil_bfs")),
            baris_perbandingan("Terpendek Jarak", hasil.get("hasil_terpendek_jarak")),
        ]

        st.dataframe(tabel_perbandingan, use_container_width=True, hide_index=True)

        # Banner "Poin utama" kondisional
        astar_rute = hasil.get("hasil_astar", {}).get("rute", [])
        jarak_hasil = hasil.get("hasil_terpendek_jarak") or {}
        jarak_rute = jarak_hasil.get("rute", [])

        if astar_rute and jarak_rute and astar_rute != jarak_rute:
            selisih = (
                jarak_hasil["total_waktu_menit"]
                - hasil["hasil_astar"]["total_waktu_menit"]
            )
            st.info(
                "💡 **Poin utama:** rute tercepat berbeda dari rute terpendek jarak. "
                f"A\\* memilih {' → '.join(astar_rute)} "
                f"({hasil['hasil_astar']['total_waktu_menit']:.2f} menit), sedangkan "
                f"rute terpendek jarak {' → '.join(jarak_rute)} "
                f"({jarak_hasil['total_waktu_menit']:.2f} menit). "
                f"Rute A\\* lebih cepat **{selisih:.2f} menit** meskipun lebih jauh."
            )
        else:
            st.info(
                "💡 **Poin utama:** pada kondisi ini rute tercepat sama dengan "
                "rute terpendek."
            )

# --- Tampilan awal sebelum pencarian ---
else:
    col_kiri, col_kanan = st.columns([3, 2])
    with col_kiri:
        st.subheader("Peta Rute")
        fig_awal = gambar_graf()
        st.pyplot(fig_awal)
        plt.close(fig_awal)
    with col_kanan:
        st.subheader("Cara Pakai")
        st.markdown("""
1. Atur **jam**, **cuaca**, dan **hari pasar** di sidebar kiri
2. Pilih **titik awal** dan **RS tujuan**
3. (Opsional) Tambahkan **laporan kejadian** (kecelakaan/banjir)
4. Klik **Cari Rute**

**Sistem akan:**
- Menentukan status ruas dengan **First Order Logic**
- Menghitung P(macet) dengan **Bayesian Network**
- Mencari rute tercepat dengan **A\\* Search**
        """)
        st.info("Graf menampilkan 5 simpul contoh: A (lokasi), B, C, D, dan RS (rumah sakit).")
