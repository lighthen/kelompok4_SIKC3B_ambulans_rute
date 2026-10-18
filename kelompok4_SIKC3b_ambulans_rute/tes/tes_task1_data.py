"""
Tes Task 1: verifikasi file data JSON valid dan koordinat admissible.
"""
import json
import math
import os
import pytest


DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")


def muat_json(nama_file):
    """Memuat file JSON dari folder data."""
    path = os.path.join(DIREKTORI_ROOT, "data", nama_file)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def jarak_lurus(koordinat_a, koordinat_b):
    """Menghitung jarak Euclidean antara dua koordinat."""
    dx = koordinat_a["x"] - koordinat_b["x"]
    dy = koordinat_a["y"] - koordinat_b["y"]
    return math.sqrt(dx**2 + dy**2)


class TestGrafContoh:
    """Tes untuk file data/graf_contoh.json."""

    def test_file_dapat_dibaca(self):
        """Graf contoh harus dapat dibaca sebagai JSON yang valid."""
        data = muat_json("graf_contoh.json")
        assert isinstance(data, dict)

    def test_jumlah_simpul(self):
        """Graf harus memiliki tepat 5 simpul."""
        data = muat_json("graf_contoh.json")
        assert len(data["simpul"]) == 5

    def test_id_simpul(self):
        """Graf harus memiliki simpul A, B, C, D, RS."""
        data = muat_json("graf_contoh.json")
        id_simpul = {s["id"] for s in data["simpul"]}
        assert id_simpul == {"A", "B", "C", "D", "RS"}

    def test_jumlah_ruas(self):
        """Graf harus memiliki tepat 5 ruas."""
        data = muat_json("graf_contoh.json")
        assert len(data["ruas"]) == 5

    def test_ruas_memiliki_atribut_lengkap(self):
        """Setiap ruas harus memiliki: id, dari, ke, jarak_km, tipe, kecepatan_kmh."""
        data = muat_json("graf_contoh.json")
        for ruas in data["ruas"]:
            assert "id" in ruas, f"Ruas {ruas} tidak punya id"
            assert "dari" in ruas, f"Ruas {ruas} tidak punya dari"
            assert "ke" in ruas, f"Ruas {ruas} tidak punya ke"
            assert "jarak_km" in ruas, f"Ruas {ruas} tidak punya jarak_km"
            assert "tipe" in ruas, f"Ruas {ruas} tidak punya tipe"
            assert "kecepatan_kmh" in ruas, f"Ruas {ruas} tidak punya kecepatan_kmh"

    def test_koordinat_admissible(self):
        """
        Syarat admissible heuristik A*:
        jarak garis lurus antar simpul <= jarak ruas yang menghubungkannya.
        Jika syarat ini dilanggar, heuristik tidak admissible dan A* tidak dijamin optimal.
        """
        data = muat_json("graf_contoh.json")
        # Buat dict koordinat
        koordinat = {s["id"]: s["koordinat"] for s in data["simpul"]}

        pelanggaran = []
        for ruas in data["ruas"]:
            id_dari = ruas["dari"]
            id_ke = ruas["ke"]
            jarak_ruas = ruas["jarak_km"]
            jarak_euclid = jarak_lurus(koordinat[id_dari], koordinat[id_ke])
            if jarak_euclid > jarak_ruas + 1e-9:  # toleransi floating point
                pelanggaran.append(
                    f"Ruas {ruas['id']}: jarak lurus {jarak_euclid:.3f} km "
                    f"> jarak ruas {jarak_ruas:.3f} km"
                )

        assert len(pelanggaran) == 0, (
            "Koordinat melanggar syarat admissible:\n" + "\n".join(pelanggaran)
        )

    def test_jarak_ruas_sesuai_spesifikasi(self):
        """Verifikasi jarak ruas sesuai tabel spesifikasi requirements.md."""
        data = muat_json("graf_contoh.json")
        jarak_harapan = {
            "A-B": 2.0,
            "B-RS": 3.0,
            "A-C": 2.5,
            "C-D": 2.0,
            "D-RS": 2.0,
        }
        jarak_aktual = {r["id"]: r["jarak_km"] for r in data["ruas"]}
        for id_ruas, jarak in jarak_harapan.items():
            assert id_ruas in jarak_aktual, f"Ruas {id_ruas} tidak ditemukan"
            assert abs(jarak_aktual[id_ruas] - jarak) < 1e-9, (
                f"Ruas {id_ruas}: jarak harapan {jarak}, aktual {jarak_aktual[id_ruas]}"
            )


class TestCptBawaan:
    """Tes untuk file data/cpt_bawaan.json."""

    def test_file_dapat_dibaca(self):
        """CPT bawaan harus dapat dibaca sebagai JSON yang valid."""
        data = muat_json("cpt_bawaan.json")
        assert isinstance(data, dict)

    def test_jumlah_entri_cpt(self):
        """CPT harus memiliki tepat 16 entri (8 kombinasi × 2 tipe ruas)."""
        data = muat_json("cpt_bawaan.json")
        assert len(data["cpt"]) == 16

    def test_probabilitas_dalam_rentang(self):
        """Semua nilai p_macet_ya harus berada di [0, 1]."""
        data = muat_json("cpt_bawaan.json")
        for entri in data["cpt"]:
            p = entri["p_macet_ya"]
            assert 0.0 <= p <= 1.0, (
                f"p_macet_ya={p} di luar rentang [0,1] untuk entri: {entri}"
            )

    def test_cpt_utama_sesuai_spesifikasi(self):
        """Verifikasi nilai CPT tipe utama sesuai tabel requirements.md."""
        data = muat_json("cpt_bawaan.json")
        # Buat lookup dict
        lookup = {
            (e["jam"], e["cuaca"], e["hari_pasar"], e["tipe_ruas"]): e["p_macet_ya"]
            for e in data["cpt"]
        }
        harapan = {
            ("Sibuk", "Hujan", "Ya", "utama"): 0.95,
            ("Sibuk", "Hujan", "Tidak", "utama"): 0.85,
            ("Sibuk", "Cerah", "Ya", "utama"): 0.80,
            ("Sibuk", "Cerah", "Tidak", "utama"): 0.60,
            ("Normal", "Hujan", "Ya", "utama"): 0.60,
            ("Normal", "Hujan", "Tidak", "utama"): 0.40,
            ("Normal", "Cerah", "Ya", "utama"): 0.35,
            ("Normal", "Cerah", "Tidak", "utama"): 0.10,
        }
        for kunci, nilai_harapan in harapan.items():
            assert kunci in lookup, f"Entri {kunci} tidak ditemukan di CPT"
            assert abs(lookup[kunci] - nilai_harapan) < 1e-9, (
                f"CPT {kunci}: harapan {nilai_harapan}, aktual {lookup[kunci]}"
            )

    def test_cpt_lokal_sesuai_rumus(self):
        """
        CPT tipe lokal harus mengikuti rumus:
        P_lokal = max(0.02, min(0.98, P_utama - 0.15))  (ASUMSI)
        """
        data = muat_json("cpt_bawaan.json")
        lookup = {
            (e["jam"], e["cuaca"], e["hari_pasar"], e["tipe_ruas"]): e["p_macet_ya"]
            for e in data["cpt"]
        }
        kombinasi = [
            ("Sibuk", "Hujan", "Ya"),
            ("Sibuk", "Hujan", "Tidak"),
            ("Sibuk", "Cerah", "Ya"),
            ("Sibuk", "Cerah", "Tidak"),
            ("Normal", "Hujan", "Ya"),
            ("Normal", "Hujan", "Tidak"),
            ("Normal", "Cerah", "Ya"),
            ("Normal", "Cerah", "Tidak"),
        ]
        for jam, cuaca, pasar in kombinasi:
            p_utama = lookup[(jam, cuaca, pasar, "utama")]
            p_lokal_harapan = max(0.02, min(0.98, p_utama - 0.15))
            p_lokal_aktual = lookup[(jam, cuaca, pasar, "lokal")]
            assert abs(p_lokal_aktual - p_lokal_harapan) < 1e-9, (
                f"CPT lokal ({jam},{cuaca},{pasar}): "
                f"harapan {p_lokal_harapan}, aktual {p_lokal_aktual}"
            )
