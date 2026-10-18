"""
fol.py — First Order Logic dengan forward chaining untuk menentukan status ruas.

Implementasi forward chaining sederhana tanpa library logika:
- Fakta disimpan sebagai set of tuples, misal: {("Kecelakaan", "B-RS")}
- Aturan R1-R7 diimplementasikan sebagai fungsi Python
- Iterasi sampai tidak ada fakta baru yang bisa disimpulkan (titik tetap)

Hanya mendukung klausa Horn dengan variabel yang diinstansiasi ke konstanta
(nama ruas dan simpul). Tidak ada variabel bebas.

Urutan eksekusi:
1. Jalankan R1-R4 berulang sampai titik tetap (inferensi positif).
2. Jalankan R5 sekali: ruas yang tidak Terblokir → DapatDilalui.
   (Asumsi Dunia Tertutup: apa yang tidak bisa dibuktikan Terblokir, dianggap DapatDilalui)
3. Jalankan R6: simpul yang semua ruasnya Terblokir → Terisolasi.
4. Jalankan R7: jika RS atau titik_awal Terisolasi → RSTidakTerjangkau.

Mengapa R5 dan R6 dijalankan setelah R1-R4 selesai:
R5 memakai negasi ("tidak Terblokir"), dan R6 memakai kuantor universal
("semua ruas terhubung"). Keduanya hanya bisa dievaluasi dengan benar
setelah semua fakta positif (R1-R4) sudah stabil di titik tetap.
Ini adalah penerapan strategi Asumsi Dunia Tertutup (Closed World Assumption).
"""

import re


# =========================================================================
# KONSTANTA PREDIKAT
# =========================================================================

# Predikat fakta lapangan (masukan dari pengguna)
KECELAKAAN = "Kecelakaan"
BANJIR = "Banjir"
TERTUTUP = "Tertutup"
LAJUR_LEBIH_DARI_SATU = "LajurTersisaLebihDariSatu"

# Predikat struktural (dari data graf, diisi oleh pipeline)
TERHUBUNG = "Terhubung"  # Terhubung(simpul, id_ruas)

# Predikat turunan
TERBLOKIR = "Terblokir"
TERBATAS = "Terbatas"
DAPAT_DILALUI = "DapatDilalui"
TERISOLASI = "Terisolasi"
RS_TIDAK_TERJANGKAU = "RSTidakTerjangkau"


# =========================================================================
# ANOTASI PENGIKATAN VARIABEL PADA JEJAK
# =========================================================================

_POLA_LANGKAH = re.compile(r"^(R\d)\s+(aktif|tidak aktif) untuk (\S+)")


def _anotasi_pengikatan(jejak: list) -> list:
    """
    Menambahkan anotasi pengikatan variabel ke tiap langkah jejak agar mudah
    dibaca: aturan berkuantor ∀r mengikat variabel r ke satu konstanta ruas
    (mis. r = B-RS), sedangkan R6 mengikat variabel simpul s.

    Catatan: ini murni anotasi tampilan. Mesin inferensi tetap bekerja dengan
    pencocokan pola ground (loop per ruas), bukan unifikasi penuh.
    Tidak mengubah fakta, status ruas, maupun hasil inferensi.
    """
    hasil = []
    for langkah in jejak:
        m = _POLA_LANGKAH.match(langkah)
        if m:
            kode, _, objek = m.groups()
            variabel = "s" if kode == "R6" else "r"
            langkah = f"{langkah}  [pengikatan: {variabel} = {objek}]"
        hasil.append(langkah)
    return hasil


# =========================================================================
# FUNGSI UTAMA
# =========================================================================

def jalankan_forward_chaining(
    fakta_awal: set,
    semua_ruas: list,
    semua_simpul: list,
    ruas_per_simpul: dict,
    titik_awal: str,
    tujuan: str,
) -> dict:
    """
    Menjalankan forward chaining dengan aturan R1-R7.

    Parameter:
        fakta_awal      : set of tuple, fakta lapangan dari BasisPengetahuan.dinamis["fakta"]
                          plus fakta Terhubung yang dibangun dari graf
        semua_ruas      : list ID ruas, misal ["A-B", "B-RS", "A-C", "C-D", "D-RS"]
        semua_simpul    : list ID simpul, misal ["A", "B", "C", "D", "RS"]
        ruas_per_simpul : dict {id_simpul: set(id_ruas)} — ruas yang menyentuh simpul
        titik_awal      : ID simpul awal perjalanan
        tujuan          : ID simpul tujuan (RS)

    Kembalian:
        dict {
            "ruas_terblokir"       : set ID ruas
            "ruas_terbatas"        : set ID ruas
            "ruas_dapat_dilalui"   : set ID ruas
            "simpul_terisolasi"    : set ID simpul
            "rs_tidak_terjangkau"  : bool
            "jejak_inferensi"      : list of str (langkah-langkah inferensi)
            "iterasi"              : int (jumlah iterasi sampai titik tetap)
        }
    """
    # Salin fakta agar tidak memodifikasi set asli
    fakta = set(fakta_awal)
    jejak = []

    # -------------------------------------------------------------------------
    # FASE 1: Jalankan R1-R4 sampai titik tetap
    # -------------------------------------------------------------------------
    iterasi = 0
    while True:
        fakta_baru = set()
        iterasi += 1

        for id_ruas in semua_ruas:
            # Baca kondisi untuk ruas ini
            ada_kecelakaan = (KECELAKAAN, id_ruas) in fakta
            ada_banjir = (BANJIR, id_ruas) in fakta
            tertutup = (TERTUTUP, id_ruas) in fakta
            lajur_lebih = (LAJUR_LEBIH_DARI_SATU, id_ruas) in fakta
            sudah_terblokir = (TERBLOKIR, id_ruas) in fakta
            sudah_terbatas = (TERBATAS, id_ruas) in fakta

            # R1: ∀r  Kecelakaan(r) ∧ Tertutup(r) → Terblokir(r)
            if ada_kecelakaan and tertutup and not sudah_terblokir:
                fakta_baru.add((TERBLOKIR, id_ruas))
                jejak.append(
                    f"R1 aktif untuk {id_ruas} "
                    f"karena Kecelakaan({id_ruas}) ∧ Tertutup({id_ruas}) "
                    f"→ Terblokir({id_ruas})"
                )

            # R2: ∀r  Banjir(r) → Terblokir(r)
            if ada_banjir and not sudah_terblokir:
                fakta_baru.add((TERBLOKIR, id_ruas))
                jejak.append(
                    f"R2 aktif untuk {id_ruas} "
                    f"karena Banjir({id_ruas}) "
                    f"→ Terblokir({id_ruas})"
                )

            # R3: ∀r  Kecelakaan(r) ∧ ¬Tertutup(r) ∧ ¬LajurTersisaLebihDariSatu(r) → Terblokir(r)
            if ada_kecelakaan and not tertutup and not lajur_lebih and not sudah_terblokir:
                fakta_baru.add((TERBLOKIR, id_ruas))
                jejak.append(
                    f"R3 aktif untuk {id_ruas} "
                    f"karena Kecelakaan({id_ruas}) ∧ ¬Tertutup({id_ruas}) "
                    f"∧ ¬LajurTersisaLebihDariSatu({id_ruas}) "
                    f"→ Terblokir({id_ruas})"
                )

            # R4: ∀r  Kecelakaan(r) ∧ ¬Tertutup(r) ∧ LajurTersisaLebihDariSatu(r) → Terbatas(r)
            if ada_kecelakaan and not tertutup and lajur_lebih and not sudah_terbatas:
                fakta_baru.add((TERBATAS, id_ruas))
                jejak.append(
                    f"R4 aktif untuk {id_ruas} "
                    f"karena Kecelakaan({id_ruas}) ∧ ¬Tertutup({id_ruas}) "
                    f"∧ LajurTersisaLebihDariSatu({id_ruas}) "
                    f"→ Terbatas({id_ruas})"
                )

        # Cek apakah ada fakta baru yang belum ada di fakta
        fakta_baru_bersih = fakta_baru - fakta
        if not fakta_baru_bersih:
            break  # Titik tetap tercapai
        fakta.update(fakta_baru_bersih)

    jejak.append(f"[Titik tetap R1-R4 tercapai setelah {iterasi} iterasi]")

    # -------------------------------------------------------------------------
    # FASE 2: R5 — Asumsi Dunia Tertutup
    # ∀r  ¬Terblokir(r) → DapatDilalui(r)
    # -------------------------------------------------------------------------
    for id_ruas in semua_ruas:
        terblokir = (TERBLOKIR, id_ruas) in fakta
        if not terblokir:
            fakta.add((DAPAT_DILALUI, id_ruas))
            jejak.append(
                f"R5 aktif untuk {id_ruas} "
                f"karena ¬Terblokir({id_ruas}) [Asumsi Dunia Tertutup] "
                f"→ DapatDilalui({id_ruas})"
            )
        else:
            jejak.append(
                f"R5 tidak aktif untuk {id_ruas} "
                f"karena Terblokir({id_ruas}) sudah terbukti"
            )

    # -------------------------------------------------------------------------
    # FASE 3: R6 — Deteksi simpul terisolasi
    # ∀s  (∀r  Terhubung(s,r) → Terblokir(r)) → Terisolasi(s)
    # -------------------------------------------------------------------------
    for id_simpul in semua_simpul:
        ruas_terhubung = ruas_per_simpul.get(id_simpul, set())
        if not ruas_terhubung:
            continue  # Simpul tanpa ruas dilewati
        semua_terblokir = all(
            (TERBLOKIR, id_ruas) in fakta
            for id_ruas in ruas_terhubung
        )
        if semua_terblokir:
            fakta.add((TERISOLASI, id_simpul))
            ruas_str = ", ".join(sorted(ruas_terhubung))
            jejak.append(
                f"R6 aktif untuk {id_simpul} "
                f"karena semua ruas terhubung [{ruas_str}] Terblokir "
                f"→ Terisolasi({id_simpul})"
            )

    # -------------------------------------------------------------------------
    # FASE 4: R7 — RS tidak terjangkau
    # Terisolasi(RS) ∨ Terisolasi(titik_awal) → RSTidakTerjangkau
    # -------------------------------------------------------------------------
    rs_terisolasi = (TERISOLASI, tujuan) in fakta
    awal_terisolasi = (TERISOLASI, titik_awal) in fakta
    rs_tidak_terjangkau = rs_terisolasi or awal_terisolasi

    if rs_tidak_terjangkau:
        fakta.add((RS_TIDAK_TERJANGKAU,))
        alasan = []
        if rs_terisolasi:
            alasan.append(f"Terisolasi({tujuan})")
        if awal_terisolasi:
            alasan.append(f"Terisolasi({titik_awal})")
        jejak.append(
            f"R7 aktif karena {' ∨ '.join(alasan)} "
            f"→ RSTidakTerjangkau"
        )
    else:
        jejak.append("R7 tidak aktif — RS dapat dijangkau")

    # -------------------------------------------------------------------------
    # Kumpulkan hasil
    # -------------------------------------------------------------------------
    # Filter hanya tuple 2-elemen (predikat, objek) agar fakta Terhubung
    # yang bertipe 3-elemen tidak menyebabkan ValueError saat unpack.
    ruas_terblokir = {t[1] for t in fakta if len(t) == 2 and t[0] == TERBLOKIR}
    ruas_terbatas = {t[1] for t in fakta if len(t) == 2 and t[0] == TERBATAS}
    ruas_dapat_dilalui = {t[1] for t in fakta if len(t) == 2 and t[0] == DAPAT_DILALUI}
    simpul_terisolasi = {t[1] for t in fakta if len(t) == 2 and t[0] == TERISOLASI}

    return {
        "ruas_terblokir": ruas_terblokir,
        "ruas_terbatas": ruas_terbatas,
        "ruas_dapat_dilalui": ruas_dapat_dilalui,
        "simpul_terisolasi": simpul_terisolasi,
        "rs_tidak_terjangkau": rs_tidak_terjangkau,
        "jejak_inferensi": _anotasi_pengikatan(jejak),
        "iterasi": iterasi,
    }


# =========================================================================
# FUNGSI BANTU — MEMBANGUN FAKTA STRUKTURAL DARI GRAF
# =========================================================================

def bangun_fakta_struktural(basis) -> tuple:
    """
    Membangun fakta Terhubung(simpul, id_ruas) dan dict ruas_per_simpul
    dari pengetahuan tetap BasisPengetahuan.

    Fakta Terhubung menyatakan bahwa simpul s adalah salah satu ujung ruas r.
    Graf tidak berarah: setiap ruas menyentuh dua simpul.

    Parameter:
        basis : BasisPengetahuan yang sudah dimuat dari file

    Kembalian:
        (fakta_struktural: set, ruas_per_simpul: dict)
        - fakta_struktural: set of tuple ("Terhubung", id_simpul, id_ruas)
        - ruas_per_simpul : dict {id_simpul: set(id_ruas)}
    """
    fakta_struktural = set()
    ruas_per_simpul = {}

    for simpul in basis.tetap["simpul"]:
        ruas_per_simpul[simpul["id"]] = set()

    for ruas in basis.tetap["ruas"]:
        id_ruas = ruas["id"]
        dari = ruas["dari"]
        ke = ruas["ke"]
        fakta_struktural.add((TERHUBUNG, dari, id_ruas))
        fakta_struktural.add((TERHUBUNG, ke, id_ruas))
        ruas_per_simpul[dari].add(id_ruas)
        ruas_per_simpul[ke].add(id_ruas)

    return fakta_struktural, ruas_per_simpul


def jalankan_fol_dari_basis(basis) -> dict:
    """
    Fungsi tingkat tinggi yang menjalankan FOL langsung dari BasisPengetahuan.

    Membaca fakta lapangan dari basis.dinamis["fakta"] dan menjalankan
    forward chaining. Hasil dikembalikan sebagai dict.

    Parameter:
        basis : BasisPengetahuan dengan fakta dan konteks yang sudah diatur

    Kembalian:
        dict hasil dari jalankan_forward_chaining()
    """
    fakta_struktural, ruas_per_simpul = bangun_fakta_struktural(basis)
    fakta_gabungan = basis.dinamis["fakta"] | fakta_struktural

    semua_ruas = [r["id"] for r in basis.tetap["ruas"]]
    semua_simpul = [s["id"] for s in basis.tetap["simpul"]]
    titik_awal = basis.dinamis.get("titik_awal") or semua_simpul[0]
    tujuan = basis.dinamis.get("tujuan") or semua_simpul[-1]

    return jalankan_forward_chaining(
        fakta_awal=fakta_gabungan,
        semua_ruas=semua_ruas,
        semua_simpul=semua_simpul,
        ruas_per_simpul=ruas_per_simpul,
        titik_awal=titik_awal,
        tujuan=tujuan,
    )
