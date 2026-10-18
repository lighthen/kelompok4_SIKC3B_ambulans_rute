"""
Tes untuk sumber/bayes.py

Mencakup:
- buat_model_bayes() berhasil dibuat
- mode_aktif() mengembalikan string yang valid
- p_macet() mengembalikan float di [0,1]
- Nilai CPT utama sesuai spesifikasi requirements.md
- Nilai CPT lokal = max(0.02, min(0.98, P_utama - 0.15))
- hitung_p_macet_semua_ruas() menghitung semua ruas
- Override P(macet) manual bekerja
- Kombinasi tidak valid raise ValueError (mode fallback)
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumber.basis_pengetahuan import BasisPengetahuan
from sumber.bayes import (
    buat_model_bayes,
    p_macet,
    p_macet_marginal,
    diagnostik_cuaca,
    prior_cuaca,
    hitung_p_macet_semua_ruas,
    mode_aktif,
    PRIOR_CUACA,
    CUACA_TIDAK_DIKETAHUI,
    _PGMPY_TERSEDIA,
)

DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_GRAF = os.path.join(DIREKTORI_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIREKTORI_ROOT, "data", "cpt_bawaan.json")


@pytest.fixture
def basis():
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    return bp


@pytest.fixture
def model(basis):
    return buat_model_bayes(basis)


# =========================================================================
# Tes umum (berlaku untuk pgmpy dan fallback)
# =========================================================================

class TestModeAktif:
    def test_mode_aktif_mengembalikan_string(self):
        """mode_aktif() harus mengembalikan string non-kosong."""
        hasil = mode_aktif()
        assert isinstance(hasil, str)
        assert len(hasil) > 0

    def test_mode_aktif_valid(self):
        """Mode harus pgmpy atau fallback."""
        hasil = mode_aktif()
        assert hasil in ("pgmpy", "fallback (dict Python)"), \
            f"Mode tidak dikenal: {hasil}"


class TestBuatModelBayes:
    def test_model_terbuat(self, model):
        """buat_model_bayes() harus menghasilkan objek non-None."""
        assert model is not None

    def test_model_terbuat_dari_basis(self, basis):
        """Model harus bisa dibuat dari BasisPengetahuan yang valid."""
        m = buat_model_bayes(basis)
        assert m is not None


class TestPMacet:
    """
    Tes nilai P(macet) berdasarkan tabel CPT requirements.md.

    Tabel CPT utama (ASUMSI):
    | Jam    | Cuaca | HariPasar | P(Macet=Ya) utama |
    | Sibuk  | Hujan | Ya        | 0.95              |
    | Sibuk  | Hujan | Tidak     | 0.85              |
    | Sibuk  | Cerah | Ya        | 0.80              |
    | Sibuk  | Cerah | Tidak     | 0.60              |
    | Normal | Hujan | Ya        | 0.60              |
    | Normal | Hujan | Tidak     | 0.40              |
    | Normal | Cerah | Ya        | 0.35              |
    | Normal | Cerah | Tidak     | 0.10              |

    Tabel lokal: P_lokal = max(0.02, min(0.98, P_utama - 0.15))
    """

    def test_hasil_adalah_float(self, model):
        """p_macet() harus mengembalikan float."""
        hasil = p_macet(model, "Sibuk", "Hujan", "Ya", "utama")
        assert isinstance(hasil, float)

    def test_hasil_dalam_rentang(self, model):
        """Semua nilai p_macet() harus di [0, 1]."""
        kombinasi = [
            ("Sibuk", "Hujan", "Ya", "utama"),
            ("Sibuk", "Hujan", "Tidak", "utama"),
            ("Normal", "Cerah", "Tidak", "lokal"),
        ]
        for jam, cuaca, pasar, tipe in kombinasi:
            hasil = p_macet(model, jam, cuaca, pasar, tipe)
            assert 0.0 <= hasil <= 1.0, \
                f"p_macet({jam},{cuaca},{pasar},{tipe}) = {hasil} di luar [0,1]"

    def test_sibuk_hujan_ya_utama(self, model):
        """P(macet | Sibuk, Hujan, Ya, utama) = 0.95 (ASUMSI)."""
        assert abs(p_macet(model, "Sibuk", "Hujan", "Ya", "utama") - 0.95) < 1e-6

    def test_sibuk_hujan_tidak_utama(self, model):
        """P(macet | Sibuk, Hujan, Tidak, utama) = 0.85 (ASUMSI)."""
        assert abs(p_macet(model, "Sibuk", "Hujan", "Tidak", "utama") - 0.85) < 1e-6

    def test_sibuk_cerah_ya_utama(self, model):
        """P(macet | Sibuk, Cerah, Ya, utama) = 0.80 (ASUMSI)."""
        assert abs(p_macet(model, "Sibuk", "Cerah", "Ya", "utama") - 0.80) < 1e-6

    def test_sibuk_cerah_tidak_utama(self, model):
        """P(macet | Sibuk, Cerah, Tidak, utama) = 0.60 (ASUMSI)."""
        assert abs(p_macet(model, "Sibuk", "Cerah", "Tidak", "utama") - 0.60) < 1e-6

    def test_normal_hujan_ya_utama(self, model):
        """P(macet | Normal, Hujan, Ya, utama) = 0.60 (ASUMSI)."""
        assert abs(p_macet(model, "Normal", "Hujan", "Ya", "utama") - 0.60) < 1e-6

    def test_normal_hujan_tidak_utama(self, model):
        """P(macet | Normal, Hujan, Tidak, utama) = 0.40 (ASUMSI)."""
        assert abs(p_macet(model, "Normal", "Hujan", "Tidak", "utama") - 0.40) < 1e-6

    def test_normal_cerah_ya_utama(self, model):
        """P(macet | Normal, Cerah, Ya, utama) = 0.35 (ASUMSI)."""
        assert abs(p_macet(model, "Normal", "Cerah", "Ya", "utama") - 0.35) < 1e-6

    def test_normal_cerah_tidak_utama(self, model):
        """P(macet | Normal, Cerah, Tidak, utama) = 0.10 (ASUMSI)."""
        assert abs(p_macet(model, "Normal", "Cerah", "Tidak", "utama") - 0.10) < 1e-6

    def test_sibuk_hujan_ya_lokal(self, model):
        """P(macet | Sibuk, Hujan, Ya, lokal) = max(0.02, min(0.98, 0.95-0.15)) = 0.80 (ASUMSI)."""
        assert abs(p_macet(model, "Sibuk", "Hujan", "Ya", "lokal") - 0.80) < 1e-6

    def test_normal_cerah_tidak_lokal(self, model):
        """P(macet | Normal, Cerah, Tidak, lokal) = max(0.02, min(0.98, 0.10-0.15)) = 0.02 (ASUMSI)."""
        assert abs(p_macet(model, "Normal", "Cerah", "Tidak", "lokal") - 0.02) < 1e-6

    def test_lokal_lebih_kecil_atau_sama_utama(self, model):
        """Untuk kombinasi yang sama, P(lokal) <= P(utama) sesuai rumus pengurangan 0.15."""
        kombinasi = [
            ("Sibuk", "Hujan", "Ya"),
            ("Sibuk", "Cerah", "Tidak"),
            ("Normal", "Cerah", "Ya"),
        ]
        for jam, cuaca, pasar in kombinasi:
            p_u = p_macet(model, jam, cuaca, pasar, "utama")
            p_l = p_macet(model, jam, cuaca, pasar, "lokal")
            assert p_l <= p_u, \
                f"P(lokal) {p_l} > P(utama) {p_u} untuk ({jam},{cuaca},{pasar})"


# =========================================================================
# Tes hitung_p_macet_semua_ruas()
# =========================================================================

class TestHitungPMacetSemuaRuas:
    def test_mengembalikan_semua_ruas(self, basis, model):
        """Harus mengembalikan P(macet) untuk semua 5 ruas."""
        basis.atur_konteks("Normal", "Cerah", "Tidak")
        hasil = hitung_p_macet_semua_ruas(basis, model)
        assert set(hasil.keys()) == {"A-B", "B-RS", "A-C", "C-D", "D-RS"}

    def test_semua_nilai_dalam_rentang(self, basis, model):
        """Semua P(macet) harus di [0, 1]."""
        basis.atur_konteks("Sibuk", "Hujan", "Ya")
        hasil = hitung_p_macet_semua_ruas(basis, model)
        for id_ruas, p in hasil.items():
            assert 0.0 <= p <= 1.0, f"P(macet)[{id_ruas}] = {p} di luar [0,1]"

    def test_ruas_utama_lebih_macet_dari_lokal_sibuk_hujan(self, basis, model):
        """Saat Sibuk+Hujan+Ya, ruas utama (A-B, B-RS) harus lebih macet dari lokal."""
        basis.atur_konteks("Sibuk", "Hujan", "Ya")
        hasil = hitung_p_macet_semua_ruas(basis, model)
        # A-B dan B-RS adalah tipe utama, A-C/C-D/D-RS adalah lokal
        assert hasil["A-B"] > hasil["A-C"]
        assert hasil["B-RS"] > hasil["C-D"]

    def test_override_manual_dipakai(self, basis, model):
        """Jika ada override manual, nilai manual yang dipakai, bukan Bayesian Network."""
        basis.atur_konteks("Sibuk", "Hujan", "Ya")
        basis.atur_p_macet_manual({"A-B": 0.42, "B-RS": 0.77})
        hasil = hitung_p_macet_semua_ruas(basis, model)
        assert abs(hasil["A-B"] - 0.42) < 1e-9
        assert abs(hasil["B-RS"] - 0.77) < 1e-9
        # Ruas lain masih dari Bayesian Network
        assert hasil["A-C"] != 0.42

    def test_tanpa_konteks_tidak_error(self, basis, model):
        """Tanpa konteks diatur, harus default ke Normal/Cerah/Tidak (tidak error)."""
        hasil = hitung_p_macet_semua_ruas(basis, model)
        assert len(hasil) == 5


# =========================================================================
# Tes khusus fallback (kombinasi tidak valid)
# =========================================================================

class TestFallbackDict:
    def test_kombinasi_tidak_valid_error(self, basis):
        """Kombinasi nilai yang tidak ada di CPT harus raise ValueError (mode fallback)."""
        if _PGMPY_TERSEDIA:
            pytest.skip("Tes ini hanya relevan untuk mode fallback")
        from sumber.bayes import _buat_lookup_dict, _cari_dict
        lookup = _buat_lookup_dict(basis)
        with pytest.raises(ValueError):
            _cari_dict(lookup, "Sangat Sibuk", "Hujan", "Ya", "utama")


# =========================================================================
# Tes marginalisasi Cuaca (cuaca tidak diketahui)
# =========================================================================

class TestMarginalisasiCuaca:
    def test_prior_cuaca_berjumlah_satu(self):
        """Prior Cuaca (ASUMSI) harus berjumlah 1 dan bernilai 0.3/0.7."""
        prior = prior_cuaca()
        assert abs(sum(prior.values()) - 1.0) < 1e-9
        assert abs(prior["Hujan"] - 0.3) < 1e-9
        assert abs(prior["Cerah"] - 0.7) < 1e-9

    def test_cuaca_tidak_diketahui_di_marginalkan(self, basis, model):
        """Cuaca 'Tidak diketahui' → nilai dihitung dengan marginalisasi."""
        basis.atur_konteks("Sibuk", CUACA_TIDAK_DIKETAHUI, "Tidak")
        hasil = hitung_p_macet_semua_ruas(basis, model)
        # utama: 0.3*0.85 + 0.7*0.60 = 0.675 ; lokal: 0.3*0.70 + 0.7*0.45 = 0.525
        assert abs(hasil["A-B"] - 0.675) < 1e-9
        assert abs(hasil["B-RS"] - 0.675) < 1e-9
        assert abs(hasil["A-C"] - 0.525) < 1e-9

    def test_marginalisasi_sesuai_rumus(self, basis, model):
        """p_macet_marginal() harus sama dengan Σ P(Macet|c)·P(c)."""
        manual = sum(
            pr * p_macet(model, "Sibuk", c, "Tidak", "utama")
            for c, pr in PRIOR_CUACA.items()
        )
        assert abs(p_macet_marginal(model, "Sibuk", "Tidak", "utama") - manual) < 1e-9

    def test_posterior_hujan_teorema_bayes(self, basis, model):
        """Posterior P(Hujan|Macet) = P(Macet|Hujan)·P(Hujan) / P(Macet)."""
        info = diagnostik_cuaca(model, "Sibuk", "Tidak", "utama")
        assert abs(info["p_macet"] - 0.675) < 1e-9
        assert abs(info["posterior"]["Hujan"] - (0.255 / 0.675)) < 1e-9
        assert abs(info["posterior"]["Cerah"] - (0.420 / 0.675)) < 1e-9
        assert abs(sum(info["posterior"].values()) - 1.0) < 1e-9

    def test_kontribusi_berjumlah_sama_dengan_total(self, basis, model):
        """Jumlah kolom kontribusi harus sama dengan P(Macet) hasil marginalisasi."""
        info = diagnostik_cuaca(model, "Sibuk", "Tidak", "lokal")
        total_kontribusi = sum(b["kontribusi"] for b in info["baris"])
        assert abs(total_kontribusi - info["p_macet"]) < 1e-9

    def test_cuaca_diketahui_tidak_dimarginalkan(self, basis, model):
        """Cuaca eksplisit tidak terpengaruh: nilai tetap murni dari CPT."""
        basis.atur_konteks("Sibuk", "Hujan", "Tidak")
        hasil = hitung_p_macet_semua_ruas(basis, model)
        assert abs(hasil["A-B"] - p_macet(model, "Sibuk", "Hujan", "Tidak", "utama")) < 1e-9
        assert abs(hasil["A-B"] - 0.85) < 1e-9
