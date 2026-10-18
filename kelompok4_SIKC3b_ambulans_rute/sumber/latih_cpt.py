"""
latih_cpt.py — Melatih CPT Bayesian Network dari dataset CSV (Fase 4, opsional).

Digunakan jika tersedia file `data/dataset_lalu_lintas.csv`.
Jika tidak ada dataset, sistem menggunakan `data/cpt_bawaan.json` (ASUMSI).

Cara pakai:
    python sumber/latih_cpt.py --input data/dataset_lalu_lintas.csv \\
                                --output data/cpt_hasil_latih.json \\
                                --kolom_jam NamaKolomJam \\
                                --kolom_cuaca NamaKolomCuaca \\
                                --kolom_pasar NamaKolomPasar \\
                                --kolom_tipe NamaKolomTipeRuas \\
                                --kolom_macet NamaKolomMacet

Catatan: Fase 4 — jalankan hanya setelah Fase 1–3 stabil.
"""

import argparse
import json
import sys


def latih_dari_csv(
    path_input: str,
    path_output: str,
    peta_kolom: dict,
) -> None:
    """
    Melatih CPT dari file CSV menggunakan pgmpy MaximumLikelihoodEstimator.

    Parameter:
        path_input  : path ke file CSV dataset lalu lintas
        path_output : path untuk menyimpan CPT hasil latihan (JSON)
        peta_kolom  : dict memetakan nama kolom dataset ke nama variabel sistem
                      contoh: {"kolom_jam": "Jam_Sibuk", "kolom_macet": "Macet"}
    """
    try:
        import pandas as pd
        from pgmpy.models import BayesianNetwork
        from pgmpy.estimators import MaximumLikelihoodEstimator
    except ImportError as e:
        print(f"Dependensi tidak tersedia: {e}")
        print("Instal dengan: pip install pandas pgmpy")
        sys.exit(1)

    print(f"Membaca dataset dari {path_input}...")
    df = pd.read_csv(path_input)

    # Ganti nama kolom sesuai peta_kolom
    df = df.rename(columns={
        peta_kolom.get("kolom_jam", "jam"): "Jam",
        peta_kolom.get("kolom_cuaca", "cuaca"): "Cuaca",
        peta_kolom.get("kolom_pasar", "hari_pasar"): "HariPasar",
        peta_kolom.get("kolom_tipe", "tipe_ruas"): "TipeRuas",
        peta_kolom.get("kolom_macet", "macet"): "Macet",
    })

    kolom_wajib = ["Jam", "Cuaca", "HariPasar", "TipeRuas", "Macet"]
    for k in kolom_wajib:
        if k not in df.columns:
            print(f"Kolom '{k}' tidak ditemukan di dataset.")
            print(f"Kolom tersedia: {list(df.columns)}")
            sys.exit(1)

    df = df[kolom_wajib].dropna()

    model = BayesianNetwork([
        ("Jam", "Macet"),
        ("Cuaca", "Macet"),
        ("HariPasar", "Macet"),
        ("TipeRuas", "Macet"),
    ])

    print("Melatih model...")
    model.fit(df, estimator=MaximumLikelihoodEstimator)

    # Ekstrak CPT ke format dict
    cpd = model.get_cpds("Macet")
    print("\nCPT Hasil Latihan:")
    print(cpd)

    # Simpan ke JSON (format kompatibel dengan cpt_bawaan.json)
    entri_cpt = []
    # (Implementasi konversi CPT pgmpy ke format dict ditambahkan sesuai kebutuhan)
    print(f"\nDisimpan ke {path_output}")
    with open(path_output, "w", encoding="utf-8") as f:
        json.dump({"cpt": entri_cpt, "_keterangan": "Hasil pelatihan dari dataset"}, f,
                  indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih CPT Bayesian Network dari CSV")
    parser.add_argument("--input", required=True, help="Path file CSV dataset")
    parser.add_argument("--output", default="data/cpt_hasil_latih.json",
                        help="Path output JSON CPT")
    parser.add_argument("--kolom_jam", default="jam")
    parser.add_argument("--kolom_cuaca", default="cuaca")
    parser.add_argument("--kolom_pasar", default="hari_pasar")
    parser.add_argument("--kolom_tipe", default="tipe_ruas")
    parser.add_argument("--kolom_macet", default="macet")
    args = parser.parse_args()

    latih_dari_csv(
        path_input=args.input,
        path_output=args.output,
        peta_kolom={
            "kolom_jam": args.kolom_jam,
            "kolom_cuaca": args.kolom_cuaca,
            "kolom_pasar": args.kolom_pasar,
            "kolom_tipe": args.kolom_tipe,
            "kolom_macet": args.kolom_macet,
        },
    )
