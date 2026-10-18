"""
bayes.py — Bayesian Network untuk menghitung P(macet) per ruas.

Struktur jaringan:
    Jam, Cuaca, HariPasar, TipeRuas  →  Macet

Implementasi otomatis memilih salah satu dari dua mode:
1. pgmpy  : menggunakan BayesianNetwork dari library pgmpy
2. fallback: pencarian tabel pada dict Python dari cpt_bawaan.json

Kedua mode mengekspos fungsi yang sama sehingga modul lain tidak perlu
tahu mode mana yang sedang dipakai.
"""

import logging

# Coba import pgmpy. Jika gagal, gunakan fallback dict Python.
try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    _PGMPY_TERSEDIA = True
except ImportError:
    _PGMPY_TERSEDIA = False

logger = logging.getLogger(__name__)


# =========================================================================
# PRIOR NODE AKAR (ASUMSI)
# =========================================================================
# ASUMSI: prior node akar Bayesian Network. Nilai 0,5/0,5 (seragam) adalah
# ASUMSI, bukan data lapangan. Prior ini hanya dipakai untuk membangun model
# pgmpy; hasil S0-S5 tidak berubah karena keempat induk selalu dijadikan
# evidence (bukan dimarjinalkan). Di mode fallback tidak ada prior.
PRIOR_AKAR = {
    "Jam": {"Sibuk": 0.5, "Normal": 0.5},
    "Cuaca": {"Hujan": 0.5, "Cerah": 0.5},
    "HariPasar": {"Ya": 0.5, "Tidak": 0.5},
    "TipeRuas": {"utama": 0.5, "lokal": 0.5},
}


def prior_node_akar() -> dict:
    """
    Mengembalikan salinan prior node akar (ASUMSI) untuk ditampilkan.

    Kembalian: dict {nama_variabel: {nilai: prior}}. Kosong jika pgmpy tidak
    tersedia (mode fallback tidak menyimpan prior).
    """
    if not _PGMPY_TERSEDIA:
        return {}
    return {nama: dict(nilai) for nama, nilai in PRIOR_AKAR.items()}


# =========================================================================
# MARGINALISASI CUACA (cuaca tidak diketahui)
# =========================================================================

# Nilai sentinel: pengguna tidak tahu cuaca saat ini.
CUACA_TIDAK_DIKETAHUI = "Tidak diketahui"

# ASUMSI: prior Cuaca saat cuaca tidak diketahui. Ini ASUMSI, bukan data
# lapangan. Dipakai HANYA untuk Hukum Probabilitas Total (marginalisasi)
# dan posterior diagnostik; S0-S5 tidak terpengaruh karena selalu memakai
# Cuaca yang eksplisit (Hujan/Cerah).
PRIOR_CUACA = {
    "Hujan": 0.3,
    "Cerah": 0.7,
}


def prior_cuaca() -> dict:
    """Mengembalikan salinan prior Cuaca (ASUMSI) untuk marginalisasi."""
    return dict(PRIOR_CUACA)


def p_macet_marginal(model: object, jam: str, hari_pasar: str, tipe_ruas: str) -> float:
    """
    Menghitung P(Macet=Ya) saat Cuaca tidak diketahui, dengan marginalisasi:

        P(Macet | jam, pasar, tipe) = Σ_c P(Macet | jam, c, pasar, tipe) · P(c)

    Cuaca adalah node akar yang independen dari induk lain pada DAG, sehingga
    Hukum Probabilitas Total ini sah. Nilai P(Macet | ...) tetap dari CPT.
    """
    total = 0.0
    for cuaca, prior in PRIOR_CUACA.items():
        total += prior * p_macet(model, jam, cuaca, hari_pasar, tipe_ruas)
    return total


def diagnostik_cuaca(model: object, jam: str, hari_pasar: str, tipe_ruas: str) -> dict:
    """
    Menyusun rincian marginalisasi + posterior diagnostik untuk satu tipe ruas.

    Posterior (Teorema Bayes):
        P(c | Macet) = P(Macet | c, ...) · P(c) / Σ_c' P(Macet | c', ...) · P(c')

    Kembalian:
        {
            "baris": [{"cuaca", "prior", "p_macet_ya", "kontribusi"}, ...],
            "p_macet": float,
            "posterior": {"Hujan": float, "Cerah": float},
        }
    """
    baris = []
    total = 0.0
    for cuaca, prior in PRIOR_CUACA.items():
        p_ya = p_macet(model, jam, cuaca, hari_pasar, tipe_ruas)
        kontribusi = prior * p_ya
        total += kontribusi
        baris.append({
            "cuaca": cuaca,
            "prior": prior,
            "p_macet_ya": p_ya,
            "kontribusi": kontribusi,
        })

    posterior = {
        b["cuaca"]: (b["kontribusi"] / total if total > 0 else 0.0)
        for b in baris
    }
    return {"baris": baris, "p_macet": total, "posterior": posterior}


# =========================================================================
# FUNGSI PUBLIK UTAMA
# =========================================================================

def buat_model_bayes(basis) -> object:
    """
    Membuat dan mengembalikan model Bayesian Network dari BasisPengetahuan.

    Jika pgmpy tersedia, model berupa BayesianNetwork pgmpy.
    Jika tidak, model berupa dict lookup dari CPT.

    Parameter:
        basis : objek BasisPengetahuan yang sudah dimuat dari file

    Kembalian:
        objek model (BayesianNetwork pgmpy atau dict lookup CPT)
    """
    if _PGMPY_TERSEDIA:
        logger.info("Bayesian Network: menggunakan pgmpy")
        return _buat_model_pgmpy(basis)
    else:
        logger.info("Bayesian Network: menggunakan fallback dict Python (pgmpy tidak tersedia)")
        return _buat_lookup_dict(basis)


def p_macet(model: object, jam: str, cuaca: str, hari_pasar: str, tipe_ruas: str) -> float:
    """
    Menghitung P(Macet=Ya) untuk satu ruas berdasarkan kondisi lingkungan.

    Interface ini sama tidak peduli apakah model menggunakan pgmpy atau fallback.

    Parameter:
        model      : objek yang dikembalikan oleh buat_model_bayes()
        jam        : "Sibuk" atau "Normal"
        cuaca      : "Hujan" atau "Cerah"
        hari_pasar : "Ya" atau "Tidak"
        tipe_ruas  : "utama" atau "lokal"

    Kembalian:
        float antara 0.0 dan 1.0

    Contoh:
        p_macet(model, "Sibuk", "Hujan", "Ya", "utama")  -> 0.95  (ASUMSI)
        p_macet(model, "Sibuk", "Hujan", "Ya", "lokal")  -> 0.80  (ASUMSI)
    """
    if _PGMPY_TERSEDIA and isinstance(model, BayesianNetwork):
        return _inferensi_pgmpy(model, jam, cuaca, hari_pasar, tipe_ruas)
    else:
        return _cari_dict(model, jam, cuaca, hari_pasar, tipe_ruas)


def hitung_p_macet_semua_ruas(basis, model: object) -> dict:
    """
    Menghitung P(macet) untuk semua ruas di basis pengetahuan menggunakan
    konteks yang sudah diatur (jam, cuaca, hari_pasar).

    Jika ada override P(macet) manual (mode validasi), nilai manual dipakai
    dan Bayesian Network diabaikan untuk ruas tersebut.

    Jika cuaca = "Tidak diketahui", nilai dihitung dengan marginalisasi
    memakai prior Cuaca (ASUMSI).

    Parameter:
        basis : BasisPengetahuan dengan konteks yang sudah diatur
        model : hasil buat_model_bayes()

    Kembalian:
        dict {id_ruas: float}
    """
    konteks = basis.dinamis["konteks"]
    jam = konteks.get("jam") or "Normal"
    cuaca = konteks.get("cuaca") or "Cerah"
    hari_pasar = konteks.get("hari_pasar") or "Tidak"
    manual = basis.dinamis.get("p_macet_manual", {})

    hasil = {}
    for ruas in basis.tetap["ruas"]:
        id_ruas = ruas["id"]
        if id_ruas in manual:
            # Mode validasi manual — abaikan Bayesian Network untuk ruas ini
            hasil[id_ruas] = manual[id_ruas]
        elif cuaca == CUACA_TIDAK_DIKETAHUI:
            # Cuaca tidak diketahui → marginalisasi dengan prior Cuaca (ASUMSI)
            hasil[id_ruas] = p_macet_marginal(model, jam, hari_pasar, ruas["tipe"])
        else:
            hasil[id_ruas] = p_macet(model, jam, cuaca, hari_pasar, ruas["tipe"])
    return hasil


def mode_aktif() -> str:
    """
    Mengembalikan string yang menjelaskan mode yang sedang dipakai.
    Berguna untuk logging dan tampilan di antarmuka.
    """
    if _PGMPY_TERSEDIA:
        return "pgmpy"
    return "fallback (dict Python)"


# =========================================================================
# IMPLEMENTASI PGMPY
# =========================================================================

def _buat_model_pgmpy(basis) -> "BayesianNetwork":
    """
    Membangun BayesianNetwork pgmpy dari CPT di BasisPengetahuan.

    Struktur:
        Jam → Macet
        Cuaca → Macet
        HariPasar → Macet
        TipeRuas → Macet
    """
    model = BayesianNetwork([
        ("Jam", "Macet"),
        ("Cuaca", "Macet"),
        ("HariPasar", "Macet"),
        ("TipeRuas", "Macet"),
    ])

    # CPD untuk variabel induk (prior seragam ASUMSI, lihat PRIOR_AKAR)
    cpd_jam = TabularCPD(
        variable="Jam", variable_card=2,
        values=[[PRIOR_AKAR["Jam"]["Sibuk"]], [PRIOR_AKAR["Jam"]["Normal"]]],
        state_names={"Jam": ["Sibuk", "Normal"]},
    )
    cpd_cuaca = TabularCPD(
        variable="Cuaca", variable_card=2,
        values=[[PRIOR_AKAR["Cuaca"]["Hujan"]], [PRIOR_AKAR["Cuaca"]["Cerah"]]],
        state_names={"Cuaca": ["Hujan", "Cerah"]},
    )
    cpd_pasar = TabularCPD(
        variable="HariPasar", variable_card=2,
        values=[[PRIOR_AKAR["HariPasar"]["Ya"]], [PRIOR_AKAR["HariPasar"]["Tidak"]]],
        state_names={"HariPasar": ["Ya", "Tidak"]},
    )
    cpd_tipe = TabularCPD(
        variable="TipeRuas", variable_card=2,
        values=[[PRIOR_AKAR["TipeRuas"]["utama"]], [PRIOR_AKAR["TipeRuas"]["lokal"]]],
        state_names={"TipeRuas": ["utama", "lokal"]},
    )

    # Bangun tabel CPD untuk Macet dari data di BasisPengetahuan
    # Urutan kolom pgmpy: TipeRuas × HariPasar × Cuaca × Jam (urutan induk terbalik)
    # State order: Jam=[Sibuk,Normal], Cuaca=[Hujan,Cerah], HariPasar=[Ya,Tidak], TipeRuas=[utama,lokal]
    lookup = _buat_lookup_dict(basis)

    # Buat nilai CPD dalam urutan yang diharapkan pgmpy
    # Iterasi: TipeRuas(0,1) × HariPasar(0,1) × Cuaca(0,1) × Jam(0,1)
    tipe_list = ["utama", "lokal"]
    pasar_list = ["Ya", "Tidak"]
    cuaca_list = ["Hujan", "Cerah"]
    jam_list = ["Sibuk", "Normal"]

    p_ya_list = []
    p_tidak_list = []
    for tipe in tipe_list:
        for pasar in pasar_list:
            for cuaca in cuaca_list:
                for jam in jam_list:
                    p = _cari_dict(lookup, jam, cuaca, pasar, tipe)
                    p_ya_list.append(p)
                    p_tidak_list.append(1.0 - p)

    cpd_macet = TabularCPD(
        variable="Macet", variable_card=2,
        values=[p_ya_list, p_tidak_list],
        evidence=["TipeRuas", "HariPasar", "Cuaca", "Jam"],
        evidence_card=[2, 2, 2, 2],
        state_names={
            "Macet": ["Ya", "Tidak"],
            "Jam": jam_list,
            "Cuaca": cuaca_list,
            "HariPasar": pasar_list,
            "TipeRuas": tipe_list,
        },
    )

    model.add_cpds(cpd_jam, cpd_cuaca, cpd_pasar, cpd_tipe, cpd_macet)

    if not model.check_model():
        raise ValueError("Model Bayesian Network tidak valid")

    return model


def _inferensi_pgmpy(
    model: "BayesianNetwork",
    jam: str,
    cuaca: str,
    hari_pasar: str,
    tipe_ruas: str,
) -> float:
    """
    Menjalankan inferensi Variable Elimination pada model pgmpy.
    Mengembalikan P(Macet=Ya) sebagai float.
    """
    inferensi = VariableElimination(model)
    hasil = inferensi.query(
        variables=["Macet"],
        evidence={
            "Jam": jam,
            "Cuaca": cuaca,
            "HariPasar": hari_pasar,
            "TipeRuas": tipe_ruas,
        },
        show_progress=False,
    )
    # Ambil nilai P(Macet=Ya) — indeks 0 sesuai state_names
    return float(hasil.values[0])


# =========================================================================
# IMPLEMENTASI FALLBACK (dict Python)
# =========================================================================

def _buat_lookup_dict(basis) -> dict:
    """
    Membuat dict lookup dari CPT di BasisPengetahuan.
    Kunci: (jam, cuaca, hari_pasar, tipe_ruas)
    Nilai: float P(Macet=Ya)
    """
    lookup = {}
    for entri in basis.tetap["cpt"]:
        kunci = (
            entri["jam"],
            entri["cuaca"],
            entri["hari_pasar"],
            entri["tipe_ruas"],
        )
        lookup[kunci] = entri["p_macet_ya"]
    return lookup


def _cari_dict(lookup: dict, jam: str, cuaca: str, hari_pasar: str, tipe_ruas: str) -> float:
    """
    Mencari nilai P(Macet=Ya) dari dict lookup.
    Raise ValueError jika kombinasi tidak ditemukan.
    """
    kunci = (jam, cuaca, hari_pasar, tipe_ruas)
    if kunci not in lookup:
        raise ValueError(
            f"Kombinasi tidak ditemukan di CPT: jam={jam}, cuaca={cuaca}, "
            f"hari_pasar={hari_pasar}, tipe_ruas={tipe_ruas}"
        )
    return lookup[kunci]
