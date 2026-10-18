"""
biaya.py — Menghitung biaya ruas sebagai waktu tempuh harapan.

Rumus:
    t_lancar   = jarak_km / kecepatan_kmh * 60          (menit)
    t_harapan  = t_lancar * (1 + p_macet * FAKTOR_MACET) (menit)

Untuk ruas Terbatas (dari FOL R4), t_harapan dikalikan lagi dengan
FAKTOR_TERBATAS.

Semua konstanta diberi nama agar mudah diubah tanpa harus mencari
angka tersebar di seluruh kode.

ASUMSI: Nilai FAKTOR_MACET = 2 dan FAKTOR_TERBATAS = 2 adalah asumsi.
"""

# ASUMSI: faktor pengali kemacetan pada rumus waktu harapan
FAKTOR_MACET: float = 2.0

# ASUMSI: faktor pengali tambahan untuk ruas yang berstatus Terbatas (FOL R4)
FAKTOR_TERBATAS: float = 2.0


def hitung_waktu_lancar(jarak_km: float, kecepatan_kmh: float) -> float:
    """
    Menghitung waktu tempuh tanpa macet dalam menit.

    Rumus: t_lancar = jarak_km / kecepatan_kmh * 60

    Parameter:
        jarak_km      : jarak ruas dalam kilometer
        kecepatan_kmh : kecepatan lancar dalam km/jam

    Kembalian:
        waktu dalam menit (float)

    Contoh:
        hitung_waktu_lancar(2.0, 40) -> 3.0 menit
    """
    if kecepatan_kmh <= 0:
        raise ValueError(f"Kecepatan harus lebih dari 0, dapat: {kecepatan_kmh}")
    return (jarak_km / kecepatan_kmh) * 60.0


def hitung_waktu_harapan(
    t_lancar: float,
    p_macet: float,
    terbatas: bool = False,
) -> float:
    """
    Menghitung waktu tempuh harapan dengan memperhitungkan probabilitas macet.

    Rumus dasar: t_harapan = t_lancar * (1 + p_macet * FAKTOR_MACET)
    Jika ruas berstatus Terbatas: t_harapan = t_harapan * FAKTOR_TERBATAS

    Parameter:
        t_lancar : waktu tempuh lancar dalam menit
        p_macet  : probabilitas macet, antara [0.0, 1.0]
        terbatas : True jika ruas berstatus Terbatas dari FOL (default: False)

    Kembalian:
        waktu harapan dalam menit (float)

    Contoh (dari tabel validasi manual requirements.md):
        hitung_waktu_harapan(3.0, 0.80)  -> 7.80 menit  (ruas A-B)
        hitung_waktu_harapan(4.5, 0.85)  -> 12.15 menit (ruas B-RS)
        hitung_waktu_harapan(3.75, 0.20) -> 5.25 menit  (ruas A-C)
        hitung_waktu_harapan(3.0, 0.15)  -> 3.90 menit  (ruas C-D)
        hitung_waktu_harapan(3.0, 0.10)  -> 3.60 menit  (ruas D-RS)
    """
    if not (0.0 <= p_macet <= 1.0):
        raise ValueError(f"p_macet harus di rentang [0, 1], dapat: {p_macet}")
    t_harapan = t_lancar * (1.0 + p_macet * FAKTOR_MACET)
    if terbatas:
        t_harapan *= FAKTOR_TERBATAS
    return t_harapan


def hitung_biaya_ruas(
    basis,
    id_ruas: str,
    p_macet: float,
    terbatas: bool = False,
) -> float:
    """
    Menghitung biaya (waktu harapan) untuk satu ruas berdasarkan data di BasisPengetahuan.

    Parameter:
        basis    : objek BasisPengetahuan yang sudah dimuat
        id_ruas  : ID ruas, misal 'A-B'
        p_macet  : probabilitas macet untuk ruas ini
        terbatas : True jika ruas berstatus Terbatas dari FOL

    Kembalian:
        waktu harapan dalam menit (float)
    """
    # Cari data ruas dari pengetahuan tetap
    data_ruas = None
    for r in basis.tetap["ruas"]:
        if r["id"] == id_ruas:
            data_ruas = r
            break

    if data_ruas is None:
        raise ValueError(f"Ruas '{id_ruas}' tidak ditemukan di basis pengetahuan")

    t_lancar = hitung_waktu_lancar(data_ruas["jarak_km"], data_ruas["kecepatan_kmh"])
    return hitung_waktu_harapan(t_lancar, p_macet, terbatas=terbatas)


def hitung_semua_biaya(
    basis,
    p_macet_per_ruas: dict,
    ruas_terbatas: set,
) -> dict:
    """
    Menghitung biaya untuk semua ruas yang tersedia (tidak terblokir).

    Parameter:
        basis             : objek BasisPengetahuan
        p_macet_per_ruas  : dict {id_ruas: float} — P(macet) per ruas
        ruas_terbatas     : set ID ruas yang berstatus Terbatas dari FOL

    Kembalian:
        dict {id_ruas: waktu_harapan_menit}
    """
    hasil = {}
    for id_ruas, p in p_macet_per_ruas.items():
        terbatas = id_ruas in ruas_terbatas
        hasil[id_ruas] = hitung_biaya_ruas(basis, id_ruas, p, terbatas=terbatas)
    return hasil
