"""
tes_smoke_aplikasi.py — Smoke test antarmuka Streamlit dengan AppTest.

Tujuan: memastikan aplikasi.py berjalan tanpa exception pada skenario S0–S5.

- S0  : validasi manual (P macet manual, tidak diekspos di UI) → render hasil.
- S1  : Normal + Cerah + bukan hari pasar.
- S2  : Sibuk + Hujan + hari pasar.
- S3  : S2 + kecelakaan B-RS (1 lajur) → R3 aktif.
- S4  : kecelakaan A-C (2 lajur) → R4 aktif.
- S5  : banjir B-RS + D-RS → RS tidak terjangkau.

AppTest hanya menjalankan skrip aplikasi tanpa browser, jadi cukup untuk
mendeteksi exception saat rendering (tab Trace A*, DAG, CPT, dll).
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from streamlit.testing.v1 import AppTest

from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.pipeline import jalankan

DIR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(DIR_ROOT, "aplikasi.py")
PATH_GRAF = os.path.join(DIR_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIR_ROOT, "data", "cpt_bawaan.json")

P_MACET_MANUAL = {
    "A-B": 0.80, "B-RS": 0.85,
    "A-C": 0.20, "C-D": 0.15, "D-RS": 0.10,
}


def _ketuk(at, label):
    """Klik tombol sidebar dengan label tertentu."""
    for b in at.sidebar.button:
        if b.label == label:
            b.click()
            return
    raise AssertionError(f"Tombol '{label}' tidak ditemukan di sidebar")


def _jalankan_ui(jam, cuaca, pasar, laporan=None, titik_awal="A", tujuan="RS"):
    """Jalankan aplikasi lewat AppTest dan klik 'Cari Rute'."""
    at = AppTest.from_file(APP, default_timeout=90)
    if laporan:
        at.session_state["laporan_kejadian"] = laporan
    at.run()
    at.sidebar.selectbox(key="jam").select(jam)
    at.sidebar.selectbox(key="cuaca").select(cuaca)
    at.sidebar.selectbox(key="hari_pasar").select(pasar)
    at.sidebar.selectbox(key="titik_awal").select(titik_awal)
    at.sidebar.selectbox(key="tujuan").select(tujuan)
    at.run()
    _ketuk(at, "Cari Rute")
    at.run()
    return at


# =========================================================================
# Aplikasi bisa dimuat tanpa exception
# =========================================================================

class TestMuat:
    def test_aplikasi_muat_tanpa_exception(self):
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        assert not at.exception, f"Exception saat memuat aplikasi: {at.exception}"

    def test_header_dan_judul(self):
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        judul = [t.value for t in at.title]
        assert "AmbuRoute" in judul


# =========================================================================
# S0 — validasi manual (render hasil manual)
# =========================================================================

class TestS0ValidasiManual:
    def test_s0_render_tanpa_exception(self):
        bp = BasisPengetahuan()
        bp.muat_dari_file(PATH_GRAF, PATH_CPT)
        bp.atur_p_macet_manual(P_MACET_MANUAL)
        bp.atur_titik("A", "RS")
        hasil = jalankan(bp)

        at = AppTest.from_file(APP, default_timeout=90)
        at.session_state["basis"] = bp
        at.session_state["hasil"] = hasil
        at.run()

        assert not at.exception, f"Exception S0: {at.exception}"
        assert hasil["hasil_astar"]["rute"] == ["A", "C", "D", "RS"]


# =========================================================================
# S1–S5 lewat interaksi UI
# =========================================================================

class TestSmokeS1:
    def test_s1(self):
        at = _jalankan_ui("Normal", "Cerah", "Tidak")
        assert not at.exception, f"Exception S1: {at.exception}"
        hasil = at.session_state["hasil"]
        assert hasil is not None
        assert hasil["hasil_astar"]["rute"] == ["A", "B", "RS"]


class TestSmokeS2:
    def test_s2(self):
        at = _jalankan_ui("Sibuk", "Hujan", "Ya")
        assert not at.exception, f"Exception S2: {at.exception}"
        hasil = at.session_state["hasil"]
        assert hasil is not None
        assert hasil["hasil_astar"]["rute"] == ["A", "B", "RS"]


class TestSmokeS3:
    def test_s3(self):
        laporan = [{"ruas": "B-RS", "jenis": "Kecelakaan",
                    "tertutup": False, "lajur_lebih": False}]
        at = _jalankan_ui("Sibuk", "Hujan", "Ya", laporan=laporan)
        assert not at.exception, f"Exception S3: {at.exception}"
        hasil = at.session_state["hasil"]
        assert "B-RS" in hasil["ruas_terblokir"]
        assert hasil["hasil_astar"]["rute"] == ["A", "C", "D", "RS"]


class TestSmokeS4:
    def test_s4(self):
        laporan = [{"ruas": "A-C", "jenis": "Kecelakaan",
                    "tertutup": False, "lajur_lebih": True}]
        at = _jalankan_ui("Normal", "Cerah", "Tidak", laporan=laporan)
        assert not at.exception, f"Exception S4: {at.exception}"
        hasil = at.session_state["hasil"]
        assert "A-C" in hasil["ruas_terbatas"]
        assert hasil["hasil_astar"]["rute"] == ["A", "B", "RS"]


class TestSmokeS5:
    def test_s5(self):
        laporan = [
            {"ruas": "B-RS", "jenis": "Banjir", "tertutup": False, "lajur_lebih": False},
            {"ruas": "D-RS", "jenis": "Banjir", "tertutup": False, "lajur_lebih": False},
        ]
        at = _jalankan_ui("Normal", "Cerah", "Tidak", laporan=laporan)
        assert not at.exception, f"Exception S5: {at.exception}"
        hasil = at.session_state["hasil"]
        assert hasil["rs_tidak_terjangkau"] is True


# =========================================================================
# TAHAP 4 — Cuaca "Tidak diketahui" (marginalisasi + posterior)
# =========================================================================

class TestSmokeCuacaTidakDiketahui:
    def test_render_tanpa_exception_dan_marginalisasi(self):
        at = _jalankan_ui("Sibuk", "Tidak diketahui", "Tidak")
        assert not at.exception, f"Exception cuaca unknown: {at.exception}"
        hasil = at.session_state["hasil"]
        assert hasil["marginalisasi_cuaca"] is not None
        utama = hasil["marginalisasi_cuaca"]["utama"]
        assert abs(utama["p_macet"] - 0.675) < 1e-9
        assert abs(utama["posterior"]["Hujan"] - (0.255 / 0.675)) < 1e-9

    def test_marginalisasi_none_saat_cuaca_diketahui(self):
        at = _jalankan_ui("Normal", "Cerah", "Tidak")
        assert not at.exception, f"Exception: {at.exception}"
        assert at.session_state["hasil"]["marginalisasi_cuaca"] is None
