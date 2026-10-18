"""
Tes untuk sumber/basis_pengetahuan.py

Mencakup:
- Memuat file berhasil
- ringkasan() mengembalikan string non-kosong
- Data simpul dan ruas sesuai JSON
- Perubahan fakta dinamis menghapus pengetahuan turunan
- reset_dinamis() bekerja benar
- ekspor_markdown() menghasilkan file
"""
import os
import pytest

# Pastikan modul sumber dapat diimpor dari folder tes
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sumber.basis_pengetahuan import BasisPengetahuan

DIREKTORI_ROOT = os.path.join(os.path.dirname(__file__), "..")
PATH_GRAF = os.path.join(DIREKTORI_ROOT, "data", "graf_contoh.json")
PATH_CPT = os.path.join(DIREKTORI_ROOT, "data", "cpt_bawaan.json")


@pytest.fixture
def bp_kosong():
    """BasisPengetahuan baru tanpa data."""
    return BasisPengetahuan()


@pytest.fixture
def bp_dimuat():
    """BasisPengetahuan yang sudah dimuat dari file."""
    bp = BasisPengetahuan()
    bp.muat_dari_file(PATH_GRAF, PATH_CPT)
    return bp


class TestMemuatFile:
    """Tes memuat data dari file JSON."""

    def test_muat_berhasil(self, bp_dimuat):
        """Memuat file tidak boleh menimbulkan error."""
        assert bp_dimuat is not None

    def test_simpul_termuat(self, bp_dimuat):
        """Harus ada 5 simpul setelah memuat."""
        assert len(bp_dimuat.tetap["simpul"]) == 5

    def test_ruas_termuat(self, bp_dimuat):
        """Harus ada 5 ruas setelah memuat."""
        assert len(bp_dimuat.tetap["ruas"]) == 5

    def test_cpt_termuat(self, bp_dimuat):
        """Harus ada 16 entri CPT setelah memuat."""
        assert len(bp_dimuat.tetap["cpt"]) == 16

    def test_aturan_fol_termuat(self, bp_dimuat):
        """Harus ada 7 aturan FOL (R1-R7)."""
        assert len(bp_dimuat.tetap["aturan_fol"]) == 7

    def test_id_simpul_benar(self, bp_dimuat):
        """ID simpul harus A, B, C, D, RS."""
        id_simpul = {s["id"] for s in bp_dimuat.tetap["simpul"]}
        assert id_simpul == {"A", "B", "C", "D", "RS"}

    def test_id_ruas_benar(self, bp_dimuat):
        """ID ruas harus A-B, B-RS, A-C, C-D, D-RS."""
        id_ruas = {r["id"] for r in bp_dimuat.tetap["ruas"]}
        assert id_ruas == {"A-B", "B-RS", "A-C", "C-D", "D-RS"}


class TestRingkasan:
    """Tes metode ringkasan()."""

    def test_ringkasan_non_kosong(self, bp_dimuat):
        """ringkasan() harus mengembalikan string non-kosong."""
        hasil = bp_dimuat.ringkasan()
        assert isinstance(hasil, str)
        assert len(hasil) > 0

    def test_ringkasan_memuat_jumlah_simpul(self, bp_dimuat):
        """ringkasan() harus menyebut '5 simpul'."""
        hasil = bp_dimuat.ringkasan()
        assert "5 simpul" in hasil

    def test_ringkasan_memuat_jumlah_ruas(self, bp_dimuat):
        """ringkasan() harus menyebut '5 ruas'."""
        hasil = bp_dimuat.ringkasan()
        assert "5 ruas" in hasil

    def test_ringkasan_turunan_belum_dihitung(self, bp_dimuat):
        """Setelah muat, turunan belum dihitung."""
        hasil = bp_dimuat.ringkasan()
        assert "Belum" in hasil


class TestFaktaDinamis:
    """Tes pengelolaan fakta dinamis dan penghapusan turunan."""

    def test_tambah_fakta(self, bp_dimuat):
        """Fakta baru harus masuk ke set fakta."""
        bp_dimuat.tambah_fakta("Kecelakaan", "B-RS")
        assert ("Kecelakaan", "B-RS") in bp_dimuat.dinamis["fakta"]

    def test_hapus_fakta(self, bp_dimuat):
        """Fakta yang dihapus harus tidak ada lagi."""
        bp_dimuat.tambah_fakta("Kecelakaan", "B-RS")
        bp_dimuat.hapus_fakta("Kecelakaan", "B-RS")
        assert ("Kecelakaan", "B-RS") not in bp_dimuat.dinamis["fakta"]

    def test_hapus_fakta_tidak_ada_tidak_error(self, bp_dimuat):
        """Menghapus fakta yang tidak ada tidak boleh error."""
        bp_dimuat.hapus_fakta("Banjir", "A-B")  # belum ditambahkan

    def test_tambah_fakta_hapus_turunan(self, bp_dimuat):
        """
        Menambahkan fakta HARUS menghapus pengetahuan turunan lama
        agar hasil inferensi sebelumnya tidak ikut terpakai.
        """
        # Simulasikan turunan sudah ada
        bp_dimuat.turunan["sudah_dihitung"] = True
        bp_dimuat.turunan["ruas_terblokir"] = {"B-RS"}

        # Tambah fakta baru
        bp_dimuat.tambah_fakta("Banjir", "A-C")

        # Turunan harus terhapus
        assert bp_dimuat.turunan["sudah_dihitung"] is False
        assert len(bp_dimuat.turunan["ruas_terblokir"]) == 0

    def test_hapus_fakta_hapus_turunan(self, bp_dimuat):
        """Menghapus fakta HARUS menghapus pengetahuan turunan lama."""
        bp_dimuat.tambah_fakta("Kecelakaan", "A-B")
        bp_dimuat.turunan["sudah_dihitung"] = True
        bp_dimuat.turunan["hasil_astar"] = {"rute": ["A", "B", "RS"]}

        bp_dimuat.hapus_fakta("Kecelakaan", "A-B")

        assert bp_dimuat.turunan["sudah_dihitung"] is False
        assert bp_dimuat.turunan["hasil_astar"] is None

    def test_atur_konteks_hapus_turunan(self, bp_dimuat):
        """Mengubah konteks HARUS menghapus pengetahuan turunan."""
        bp_dimuat.turunan["sudah_dihitung"] = True
        bp_dimuat.atur_konteks("Sibuk", "Hujan", "Ya")
        assert bp_dimuat.turunan["sudah_dihitung"] is False

    def test_atur_konteks_tersimpan(self, bp_dimuat):
        """Konteks yang diatur harus tersimpan dengan benar."""
        bp_dimuat.atur_konteks("Sibuk", "Hujan", "Ya")
        assert bp_dimuat.dinamis["konteks"]["jam"] == "Sibuk"
        assert bp_dimuat.dinamis["konteks"]["cuaca"] == "Hujan"
        assert bp_dimuat.dinamis["konteks"]["hari_pasar"] == "Ya"

    def test_atur_titik(self, bp_dimuat):
        """Titik awal dan tujuan harus tersimpan."""
        bp_dimuat.atur_titik("A", "RS")
        assert bp_dimuat.dinamis["titik_awal"] == "A"
        assert bp_dimuat.dinamis["tujuan"] == "RS"

    def test_atur_p_macet_manual(self, bp_dimuat):
        """P(macet) manual harus tersimpan di dinamis."""
        manual = {"A-B": 0.80, "B-RS": 0.85}
        bp_dimuat.atur_p_macet_manual(manual)
        assert bp_dimuat.dinamis["p_macet_manual"]["A-B"] == 0.80
        assert bp_dimuat.dinamis["p_macet_manual"]["B-RS"] == 0.85


class TestResetDinamis:
    """Tes metode reset_dinamis()."""

    def test_reset_menghapus_fakta(self, bp_dimuat):
        """reset_dinamis() harus mengosongkan fakta."""
        bp_dimuat.tambah_fakta("Banjir", "A-B")
        bp_dimuat.reset_dinamis()
        assert len(bp_dimuat.dinamis["fakta"]) == 0

    def test_reset_menghapus_konteks(self, bp_dimuat):
        """reset_dinamis() harus mengosongkan konteks."""
        bp_dimuat.atur_konteks("Sibuk", "Hujan", "Ya")
        bp_dimuat.reset_dinamis()
        assert bp_dimuat.dinamis["konteks"]["jam"] is None

    def test_reset_menghapus_turunan(self, bp_dimuat):
        """reset_dinamis() harus menghapus pengetahuan turunan."""
        bp_dimuat.turunan["sudah_dihitung"] = True
        bp_dimuat.reset_dinamis()
        assert bp_dimuat.turunan["sudah_dihitung"] is False

    def test_reset_tidak_hapus_tetap(self, bp_dimuat):
        """reset_dinamis() TIDAK boleh menghapus pengetahuan tetap."""
        bp_dimuat.reset_dinamis()
        assert len(bp_dimuat.tetap["simpul"]) == 5
        assert len(bp_dimuat.tetap["ruas"]) == 5


class TestSimpanTurunan:
    """Tes metode simpan_turunan()."""

    def test_simpan_turunan(self, bp_dimuat):
        """simpan_turunan() harus menyimpan semua data turunan."""
        bp_dimuat.simpan_turunan(
            ruas_terblokir={"B-RS"},
            ruas_terbatas={"A-C"},
            rs_tidak_terjangkau=False,
            jejak_inferensi=["R3 aktif untuk B-RS"],
            p_macet_per_ruas={"A-B": 0.80},
            biaya_per_ruas={"A-B": 7.80},
            hasil_astar={"rute": ["A", "C", "D", "RS"], "total_waktu_menit": 12.75,
                         "total_jarak_km": 6.5, "jumlah_diekspansi": 4, "waktu_komputasi_detik": 0.001},
            hasil_bfs=None,
            hasil_dijkstra=None,
            hasil_terpendek_jarak=None,
        )
        assert bp_dimuat.turunan["sudah_dihitung"] is True
        assert "B-RS" in bp_dimuat.turunan["ruas_terblokir"]
        assert "A-C" in bp_dimuat.turunan["ruas_terbatas"]
        assert bp_dimuat.turunan["jejak_inferensi"] == ["R3 aktif untuk B-RS"]


class TestEksporMarkdown:
    """Tes metode ekspor_markdown()."""

    def test_file_terbentuk(self, bp_dimuat, tmp_path):
        """ekspor_markdown() harus menghasilkan file."""
        path_output = str(tmp_path / "basis_pengetahuan.md")
        bp_dimuat.ekspor_markdown(path_output)
        assert os.path.exists(path_output)

    def test_isi_memuat_simpul(self, bp_dimuat, tmp_path):
        """File ekspor harus memuat data simpul."""
        path_output = str(tmp_path / "basis_pengetahuan.md")
        bp_dimuat.ekspor_markdown(path_output)
        with open(path_output, "r", encoding="utf-8") as f:
            isi = f.read()
        assert "Lokasi Kejadian" in isi
        assert "Rumah Sakit" in isi

    def test_isi_memuat_aturan_fol(self, bp_dimuat, tmp_path):
        """File ekspor harus memuat aturan FOL R1-R7."""
        path_output = str(tmp_path / "basis_pengetahuan.md")
        bp_dimuat.ekspor_markdown(path_output)
        with open(path_output, "r", encoding="utf-8") as f:
            isi = f.read()
        for kode in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
            assert kode in isi, f"Aturan {kode} tidak ditemukan di ekspor"

    def test_isi_memuat_label_asumsi(self, bp_dimuat, tmp_path):
        """File ekspor harus memuat label ASUMSI pada CPT."""
        path_output = str(tmp_path / "basis_pengetahuan.md")
        bp_dimuat.ekspor_markdown(path_output)
        with open(path_output, "r", encoding="utf-8") as f:
            isi = f.read()
        assert "ASUMSI" in isi
