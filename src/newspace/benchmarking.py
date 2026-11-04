"""Multi-dimensional Data Mining & Persistence Benchmarking Engine.

Covers three rigorous engineering dimensions on 100% real aerospace datasets:
1. Storage & Query Engine Benchmark: SQLite (Unindexed vs Indexed) vs Pandas vs Polars Lazy vs DuckDB OLAP
2. Unsupervised Data Mining Benchmark: Clustering 7,169 NASA NHATS Accessible Asteroids
3. Supervised Actuarial Data Mining Benchmark: Leak-free Historical Launch Outcome & Risk Calibration
"""

from __future__ import annotations

from contextlib import closing
import gc
import sqlite3
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import (
    AgglomerativeClustering,
    BisectingKMeans,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    log_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from newspace.config import DATA_PROCESSED_DIR, DB_PATH
from newspace.database import SCHEMA_DDL
from newspace.ingestion import MinedDatasets


@dataclass
class BenchmarkSuiteResults:
    """Structured results across all three benchmarking dimensions."""

    engine_workloads: pd.DataFrame
    engine_summary: pd.DataFrame
    storage_footprint: pd.DataFrame
    clustering_benchmark: pd.DataFrame
    clustered_nhats: pd.DataFrame
    classification_benchmark: pd.DataFrame
    feature_importances: pd.DataFrame


def _measure_callable(fn, repeats: int = 7) -> dict[str, float]:
    """Run 1 warmup + `repeats` timed executions, capturing latency distribution and peak memory."""
    fn()  # Warmup
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    timings_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(timings_ms, dtype=np.float64)
    return {
        "median_ms": round(float(np.median(arr)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "peak_mem_kb": round(peak_bytes / 1024.0, 1),
    }


def benchmark_data_engines(
    datasets: MinedDatasets,
    indexed_db_path: Path = DB_PATH,
    processed_dir: Path = DATA_PROCESSED_DIR,
    repeats: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Benchmark 5 persistence/analytical engines across 3 domain workloads."""
    total_rows = (
        len(datasets.satellites)
        + len(datasets.asteroids)
        + len(datasets.launches)
        + len(datasets.launchers)
        + len(datasets.corridors)
    )

    # Create temporary unindexed SQLite DB to quantify exact speedup of Fase 1/2 relational indexing
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        unindexed_db = Path(tmpdir) / "unindexed.db"
        with closing(sqlite3.connect(unindexed_db)) as conn_u:
            conn_u.executescript(SCHEMA_DDL)
            datasets.launchers.to_sql("VEICULO_LANCADOR", conn_u, if_exists="append", index=False)
            datasets.launches.to_sql("HISTORICO_LANCAMENTO", conn_u, if_exists="append", index=False)
            datasets.corridors.to_sql("CORREDOR_ORBITAL", conn_u, if_exists="append", index=False)
            datasets.satellites.to_sql("OBJETO_TRAFEGO_ORBITAL", conn_u, if_exists="append", index=False)
            datasets.asteroids.to_sql("ASTEROIDE_PROSPECCAO", conn_u, if_exists="append", index=False)
            conn_u.commit()

        conn_unidx = sqlite3.connect(unindexed_db)
        conn_idx = sqlite3.connect(indexed_db_path)
        duck_conn = duckdb.connect(":memory:")

        sat_pq = (processed_dir / "objeto_trafego_orbital.parquet").as_posix()
        cor_pq = (processed_dir / "corredor_orbital.parquet").as_posix()
        ast_pq = (processed_dir / "asteroide_prospeccao.parquet").as_posix()
        lan_pq = (processed_dir / "historico_lancamento.parquet").as_posix()
        vei_pq = (processed_dir / "veiculo_lancador.parquet").as_posix()

        # SQL Queries shared across SQLite Unindexed, SQLite Indexed, and DuckDB
        sql_w1 = """
            SELECT o.id_corredor, c.regime_orbital,
                   COUNT(*) AS n_obj,
                   SUM(o.ativo) AS n_ativos,
                   SUM(o.eh_detrito_ou_estagio) AS n_detritos,
                   AVG(o.apogeu_km) AS apogeu_medio
            FROM {sat} o
            JOIN {cor} c ON o.id_corredor = c.id_corredor
            WHERE o.em_orbita = 1 AND o.ano_lancamento >= 2000
            GROUP BY o.id_corredor, c.regime_orbital
            ORDER BY n_obj DESC
        """

        sql_w2 = """
            SELECT grupo_mineralogico,
                   COUNT(*) AS n_alvos,
                   AVG(min_delta_v_kms) AS dv_medio,
                   MIN(min_delta_v_kms) AS dv_min,
                   AVG(duracao_min_dv_dias) AS dur_media
            FROM {ast}
            WHERE acessivel_nhats = 1
              AND min_delta_v_kms <= 7.0
              AND codigo_incerteza_occ <= 3
              AND num_observacoes >= 40
            GROUP BY grupo_mineralogico
            ORDER BY n_alvos DESC
        """

        sql_w3 = """
            SELECT h.ano_lancamento,
                   COUNT(*) AS lancamentos,
                   SUM(h.sucesso_pleno) AS sucessos,
                   SUM(h.qtd_cargas) AS cargas_implantadas,
                   AVG(v.wilson_lower_95_pct) AS wilson_medio
            FROM {lan} h
            JOIN {vei} v ON h.id_veiculo = v.id_veiculo
            WHERE v.total_lancamentos >= 10 AND h.ano_lancamento >= 1990
            GROUP BY h.ano_lancamento
            ORDER BY h.ano_lancamento
        """

        sat_pd = datasets.satellites
        cor_pd = datasets.corridors
        ast_pd = datasets.asteroids
        lan_pd = datasets.launches
        vei_pd = datasets.launchers

        workloads = [
            (
                "W1_Join_Trafego_Orbital",
                len(sat_pd),
                {
                    "SQLite (Sem Índices)": lambda: pd.read_sql_query(
                        sql_w1.format(sat="OBJETO_TRAFEGO_ORBITAL", cor="CORREDOR_ORBITAL"), conn_unidx
                    ),
                    "SQLite (Indexado + WAL)": lambda: pd.read_sql_query(
                        sql_w1.format(sat="OBJETO_TRAFEGO_ORBITAL", cor="CORREDOR_ORBITAL"), conn_idx
                    ),
                    "Pandas (In-Memory)": lambda: (
                        sat_pd[(sat_pd["em_orbita"] == 1) & (sat_pd["ano_lancamento"] >= 2000)]
                        .merge(cor_pd[["id_corredor", "regime_orbital"]], on="id_corredor", how="inner")
                        .groupby(["id_corredor", "regime_orbital"], as_index=False)
                        .agg(
                            n_obj=("norad_id", "count"),
                            n_ativos=("ativo", "sum"),
                            n_detritos=("eh_detrito_ou_estagio", "sum"),
                            apogeu_medio=("apogeu_km", "mean"),
                        )
                        .sort_values("n_obj", ascending=False)
                    ),
                    "Polars (Lazy Parquet)": lambda: (
                        pl.scan_parquet(sat_pq)
                        .filter((pl.col("em_orbita") == 1) & (pl.col("ano_lancamento") >= 2000))
                        .join(pl.scan_parquet(cor_pq).select(["id_corredor", "regime_orbital"]), on="id_corredor")
                        .group_by(["id_corredor", "regime_orbital"])
                        .agg(
                            pl.len().alias("n_obj"),
                            pl.col("ativo").sum().alias("n_ativos"),
                            pl.col("eh_detrito_ou_estagio").sum().alias("n_detritos"),
                            pl.col("apogeu_km").mean().alias("apogeu_medio"),
                        )
                        .sort("n_obj", descending=True)
                        .collect()
                    ),
                    "DuckDB (OLAP Parquet)": lambda: duck_conn.execute(
                        sql_w1.format(sat=f"'{sat_pq}'", cor=f"'{cor_pq}'")
                    ).df(),
                },
            ),
            (
                "W2_Prospeccao_Asteroides_NHATS",
                len(ast_pd),
                {
                    "SQLite (Sem Índices)": lambda: pd.read_sql_query(
                        sql_w2.format(ast="ASTEROIDE_PROSPECCAO"), conn_unidx
                    ),
                    "SQLite (Indexado + WAL)": lambda: pd.read_sql_query(
                        sql_w2.format(ast="ASTEROIDE_PROSPECCAO"), conn_idx
                    ),
                    "Pandas (In-Memory)": lambda: (
                        ast_pd[
                            (ast_pd["acessivel_nhats"] == 1)
                            & (ast_pd["min_delta_v_kms"] <= 7.0)
                            & (ast_pd["codigo_incerteza_occ"] <= 3)
                            & (ast_pd["num_observacoes"] >= 40)
                        ]
                        .groupby("grupo_mineralogico", as_index=False)
                        .agg(
                            n_alvos=("spkid", "count"),
                            dv_medio=("min_delta_v_kms", "mean"),
                            dv_min=("min_delta_v_kms", "min"),
                            dur_media=("duracao_min_dv_dias", "mean"),
                        )
                        .sort_values("n_alvos", ascending=False)
                    ),
                    "Polars (Lazy Parquet)": lambda: (
                        pl.scan_parquet(ast_pq)
                        .filter(
                            (pl.col("acessivel_nhats") == 1)
                            & (pl.col("min_delta_v_kms") <= 7.0)
                            & (pl.col("codigo_incerteza_occ") <= 3)
                            & (pl.col("num_observacoes") >= 40)
                        )
                        .group_by("grupo_mineralogico")
                        .agg(
                            pl.len().alias("n_alvos"),
                            pl.col("min_delta_v_kms").mean().alias("dv_medio"),
                            pl.col("min_delta_v_kms").min().alias("dv_min"),
                            pl.col("duracao_min_dv_dias").mean().alias("dur_media"),
                        )
                        .sort("n_alvos", descending=True)
                        .collect()
                    ),
                    "DuckDB (OLAP Parquet)": lambda: duck_conn.execute(
                        sql_w2.format(ast=f"'{ast_pq}'")
                    ).df(),
                },
            ),
            (
                "W3_Serie_Historica_Atuarial",
                len(lan_pd),
                {
                    "SQLite (Sem Índices)": lambda: pd.read_sql_query(
                        sql_w3.format(lan="HISTORICO_LANCAMENTO", vei="VEICULO_LANCADOR"), conn_unidx
                    ),
                    "SQLite (Indexado + WAL)": lambda: pd.read_sql_query(
                        sql_w3.format(lan="HISTORICO_LANCAMENTO", vei="VEICULO_LANCADOR"), conn_idx
                    ),
                    "Pandas (In-Memory)": lambda: (
                        lan_pd[lan_pd["ano_lancamento"] >= 1990]
                        .merge(
                            vei_pd[vei_pd["total_lancamentos"] >= 10][["id_veiculo", "wilson_lower_95_pct"]],
                            on="id_veiculo",
                            how="inner",
                        )
                        .groupby("ano_lancamento", as_index=False)
                        .agg(
                            lancamentos=("launch_tag", "count"),
                            sucessos=("sucesso_pleno", "sum"),
                            cargas_implantadas=("qtd_cargas", "sum"),
                            wilson_medio=("wilson_lower_95_pct", "mean"),
                        )
                        .sort_values("ano_lancamento")
                    ),
                    "Polars (Lazy Parquet)": lambda: (
                        pl.scan_parquet(lan_pq)
                        .filter(pl.col("ano_lancamento") >= 1990)
                        .join(
                            pl.scan_parquet(vei_pq)
                            .filter(pl.col("total_lancamentos") >= 10)
                            .select(["id_veiculo", "wilson_lower_95_pct"]),
                            on="id_veiculo",
                        )
                        .group_by("ano_lancamento")
                        .agg(
                            pl.len().alias("lancamentos"),
                            pl.col("sucesso_pleno").sum().alias("sucessos"),
                            pl.col("qtd_cargas").sum().alias("cargas_implantadas"),
                            pl.col("wilson_lower_95_pct").mean().alias("wilson_medio"),
                        )
                        .sort("ano_lancamento")
                        .collect()
                    ),
                    "DuckDB (OLAP Parquet)": lambda: duck_conn.execute(
                        sql_w3.format(lan=f"'{lan_pq}'", vei=f"'{vei_pq}'")
                    ).df(),
                },
            ),
        ]

        rows = []
        for w_name, n_scanned, impls in workloads:
            for eng_name, fn in impls.items():
                stats = _measure_callable(fn, repeats=repeats)
                throughput_krows_s = round((n_scanned / (stats["median_ms"] / 1000.0)) / 1000.0, 1)
                rows.append(
                    {
                        "workload": w_name,
                        "engine": eng_name,
                        "registros_avaliados": n_scanned,
                        **stats,
                        "throughput_krows_s": throughput_krows_s,
                    }
                )

        unidx_size_mb = round(unindexed_db.stat().st_size / (1024 * 1024), 3)
        conn_unidx.close()
        conn_idx.close()
        duck_conn.close()

    workloads_df = pd.DataFrame(rows)

    # Compute overall summary across all 3 workloads per engine
    summary_df = (
        workloads_df.groupby("engine", as_index=False)
        .agg(
            latencia_total_mediana_ms=("median_ms", "sum"),
            latencia_media_ms=("mean_ms", "mean"),
            p95_medio_ms=("p95_ms", "mean"),
            pico_memoria_max_kb=("peak_mem_kb", "max"),
            throughput_medio_krows_s=("throughput_krows_s", "mean"),
        )
        .sort_values("latencia_total_mediana_ms", ascending=True)
        .reset_index(drop=True)
    )
    baseline_ms = float(
        summary_df.loc[
            summary_df["engine"] == "SQLite (Sem Índices)", "latencia_total_mediana_ms"
        ].iloc[0]
    )
    summary_df["speedup_vs_sqlite_sem_indice"] = np.round(
        baseline_ms / summary_df["latencia_total_mediana_ms"], 2
    )
    summary_df["latencia_total_mediana_ms"] = summary_df["latencia_total_mediana_ms"].round(3)
    summary_df["latencia_media_ms"] = summary_df["latencia_media_ms"].round(3)
    summary_df["p95_medio_ms"] = summary_df["p95_medio_ms"].round(3)
    summary_df["throughput_medio_krows_s"] = summary_df["throughput_medio_krows_s"].round(1)

    # Storage Footprint comparison (Raw files vs SQLite Unindexed vs SQLite Indexed vs Parquet Snappy)
    idx_size_mb = round(indexed_db_path.stat().st_size / (1024 * 1024), 3)
    parquet_bytes = sum(
        (processed_dir / f).stat().st_size
        for f in [
            "veiculo_lancador.parquet",
            "historico_lancamento.parquet",
            "corredor_orbital.parquet",
            "objeto_trafego_orbital.parquet",
            "asteroide_prospeccao.parquet",
        ]
    )
    parquet_mb = round(parquet_bytes / (1024 * 1024), 3)
    raw_mb = round(
        sum(d["size_bytes"] for d in datasets.provenance["datasets"]) / (1024 * 1024), 3
    )

    storage_df = pd.DataFrame(
        [
            {
                "formato_persistencia": "Arquivos Brutos Originais (JSON + CSV + TSV)",
                "tamanho_mb": raw_mb,
                "registros_totais": total_rows,
                "razao_compressao_vs_bruto": 1.00,
            },
            {
                "formato_persistencia": "SQLite Relacional Normalizado (Sem Índices)",
                "tamanho_mb": unidx_size_mb,
                "registros_totais": total_rows,
                "razao_compressao_vs_bruto": round(raw_mb / max(unidx_size_mb, 0.001), 2),
            },
            {
                "formato_persistencia": "SQLite Relacional Normalizado + Índices B-Tree",
                "tamanho_mb": idx_size_mb,
                "registros_totais": total_rows,
                "razao_compressao_vs_bruto": round(raw_mb / max(idx_size_mb, 0.001), 2),
            },
            {
                "formato_persistencia": "Apache Parquet Colunar (Compressão ZSTD/Snappy)",
                "tamanho_mb": parquet_mb,
                "registros_totais": total_rows,
                "razao_compressao_vs_bruto": round(raw_mb / max(parquet_mb, 0.001), 2),
            },
        ]
    )
    return workloads_df, summary_df, storage_df


def benchmark_clustering_mining(
    asteroids_df: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark 5 Unsupervised Clustering algorithms on all 7,169 real NASA NHATS asteroids."""
    nhats = asteroids_df[asteroids_df["acessivel_nhats"] == 1].copy()
    feature_cols = [
        "min_delta_v_kms",
        "duracao_min_dv_dias",
        "mag_absoluta_h",
        "excentricidade_e",
        "semieixo_maior_au",
        "inclinacao_graus",
        "moid_au",
    ]
    nhats["log_observacoes"] = np.log1p(nhats["num_observacoes"].fillna(0.0))
    all_feats = feature_cols + ["log_observacoes"]

    clean = nhats.dropna(subset=all_feats).copy().reset_index(drop=True)
    X = RobustScaler().fit_transform(clean[all_feats].to_numpy(dtype=np.float64))

    models = {
        "K-Means (Elkan, k=4)": KMeans(n_clusters=4, n_init=10, random_state=random_state),
        "MiniBatch K-Means (k=4)": MiniBatchKMeans(
            n_clusters=4, batch_size=512, n_init=10, random_state=random_state
        ),
        "Bisecting K-Means (k=4)": BisectingKMeans(n_clusters=4, random_state=random_state),
        "Gaussian Mixture (GMM Full, k=4)": GaussianMixture(
            n_components=4, covariance_type="full", random_state=random_state
        ),
        "Agglomerative Ward (k=4)": AgglomerativeClustering(n_clusters=4, linkage="ward"),
    }

    bench_rows = []
    best_labels = None
    best_sil = -2.0
    best_algo_name = ""

    for name, model in models.items():
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        labels = model.fit_predict(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        sil = float(silhouette_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))

        if sil > best_sil:
            best_sil = sil
            best_labels = labels
            best_algo_name = name

        bench_rows.append(
            {
                "algoritmo": name,
                "amostras_nhats": len(clean),
                "silhouette_score": round(sil, 4),
                "calinski_harabasz": round(ch, 1),
                "davies_bouldin": round(dbi, 4),
                "tempo_execucao_ms": round(elapsed_ms, 2),
                "pico_memoria_kb": round(peak_bytes / 1024.0, 1),
            }
        )

    bench_df = (
        pd.DataFrame(bench_rows)
        .sort_values("silhouette_score", ascending=False)
        .reset_index(drop=True)
    )

    # Label physical cluster archetypes ordered by median Delta-V and Inclination
    clean["cluster_id"] = best_labels
    clean["algoritmo_vencedor"] = best_algo_name
    cluster_stats = (
        clean.groupby("cluster_id")
        .agg(
            med_dv=("min_delta_v_kms", "median"),
            med_obs=("num_observacoes", "median"),
            med_e=("excentricidade_e", "median"),
        )
        .sort_values("med_dv")
    )
    ordered_ids = list(cluster_stats.index)
    archetype_names = [
        "Cluster A: Órbita Quase-Terrestre (Baixo Δv / Alta Acessibilidade)",
        "Cluster B: NEAs Moderados Bem Caracterizados ( Janela Comercial )",
        "Cluster C: NEAs Energéticos / Excentricidade Moderada-Alta",
        "Cluster D: Alvos de Alta Energia / Excentricidade Extrema",
    ]
    label_map = {
        cid: archetype_names[min(idx, len(archetype_names) - 1)]
        for idx, cid in enumerate(ordered_ids)
    }
    clean["arquetipo_astrodinamico"] = clean["cluster_id"].map(label_map)
    return bench_df, clean


def _build_leak_free_actuarial_features(launches_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct strictly chronological pre-launch features (zero look-ahead leakage)."""
    df = launches_df.sort_values(["data_lancamento", "launch_tag"]).reset_index(drop=True).copy()

    # Cumulative flights & successes of vehicle BEFORE current launch
    df["voos_previos_veiculo"] = df.groupby("id_veiculo").cumcount()
    cum_succ_v = df.groupby("id_veiculo")["sucesso_pleno"].cumsum() - df["sucesso_pleno"]
    # Prior smoothed success rate (Laplace/Beta prior 3.6/4.0 = 0.90 neutral aerospace baseline)
    df["confiabilidade_previa_veiculo"] = (cum_succ_v + 3.6) / (df["voos_previos_veiculo"] + 4.0)

    first_year_v = df.groupby("id_veiculo")["ano_lancamento"].transform("min")
    df["maturidade_veiculo_anos"] = np.maximum(0, df["ano_lancamento"] - first_year_v)

    df["voos_previos_agencia"] = df.groupby("agencia_operador").cumcount()
    cum_succ_a = df.groupby("agencia_operador")["sucesso_pleno"].cumsum() - df["sucesso_pleno"]
    df["confiabilidade_previa_agencia"] = (cum_succ_a + 3.6) / (df["voos_previos_agencia"] + 4.0)

    df["voos_previos_sitio"] = df.groupby("sitio_lancamento").cumcount()
    df["log_qtd_cargas"] = np.log1p(df["qtd_cargas"])
    df["log_voos_veiculo"] = np.log1p(df["voos_previos_veiculo"])
    df["log_voos_agencia"] = np.log1p(df["voos_previos_agencia"])

    feature_names = [
        "ano_lancamento",
        "log_voos_veiculo",
        "confiabilidade_previa_veiculo",
        "maturidade_veiculo_anos",
        "log_voos_agencia",
        "confiabilidade_previa_agencia",
        "voos_previos_sitio",
        "espaco_profundo",
        "log_qtd_cargas",
    ]
    X = df[feature_names].to_numpy(dtype=np.float64)
    # Target: Mission Loss / Partial Failure Risk Event (1 = sinistro/falha, 0 = sucesso pleno)
    y_risk = (1 - df["sucesso_pleno"]).to_numpy(dtype=np.int32)
    return X, y_risk, feature_names


def benchmark_actuarial_classifiers(
    launches_df: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark 5 Supervised Classifiers for Aerospace Actuarial Sinistro Prediction (5-Fold CV)."""
    X, y_risk, feature_names = _build_leak_free_actuarial_features(launches_df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    classifiers = {
        "HistGradientBoosting (GBDT)": HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.06, max_iter=200, random_state=random_state
        ),
        "Random Forest (200 Árvores)": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        ),
        "Extra Trees (200 Árvores)": ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        ),
        "Regressão Logística L2 (Baseline Atuarial)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=random_state)),
            ]
        ),
        "Rede Neural MLP (64x32 ReLU)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=250,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    rows = []
    for name, clf in classifiers.items():
        t0 = time.perf_counter()
        probs = cross_val_predict(clf, X, y_risk, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        preds = (probs >= 0.5).astype(int)

        rows.append(
            {
                "modelo_atuarial": name,
                "lancamentos_avaliados": len(y_risk),
                "taxa_base_sinistro_pct": round(100.0 * float(y_risk.mean()), 2),
                "roc_auc": round(float(roc_auc_score(y_risk, probs)), 4),
                "pr_auc": round(float(average_precision_score(y_risk, probs)), 4),
                "brier_score_calibracao": round(float(brier_score_loss(y_risk, probs)), 5),
                "log_loss": round(float(log_loss(y_risk, probs)), 4),
                "f1_weighted": round(float(f1_score(y_risk, preds, average="weighted")), 4),
                "tempo_cv_5fold_ms": round(elapsed_ms, 1),
            }
        )

    bench_df = (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "brier_score_calibracao"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Extract permutation/tree feature importances from Random Forest on full historical dataset
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=random_state
    )
    rf.fit(X, y_risk)
    human_labels = {
        "ano_lancamento": "Era Tecnológica (Ano do Lançamento)",
        "log_voos_veiculo": "Herança de Voo Acumulada do Foguete",
        "confiabilidade_previa_veiculo": "Taxa Histórica Prévia do Foguete (Bayes)",
        "maturidade_veiculo_anos": "Idade Operacional do Modelo (Anos)",
        "log_voos_agencia": "Experiência Acumulada da Agência/Operador",
        "confiabilidade_previa_agencia": "Histórico Prévio da Agência/Operador",
        "voos_previos_sitio": "Maturidade da Base de Lançamento",
        "espaco_profundo": "Regime Energético (Escape / Espaço Profundo)",
        "log_qtd_cargas": "Complexidade de Dispensação (Nº de Cargas)",
    }
    imp_df = (
        pd.DataFrame(
            {
                "feature_codigo": feature_names,
                "fator_risco": [human_labels[f] for f in feature_names],
                "importancia_relativa_pct": np.round(100.0 * rf.feature_importances_, 2),
            }
        )
        .sort_values("importancia_relativa_pct", ascending=False)
        .reset_index(drop=True)
    )
    return bench_df, imp_df


def run_complete_benchmark_suite(datasets: MinedDatasets) -> BenchmarkSuiteResults:
    """Execute all 3 benchmarking dimensions end-to-end."""
    workloads_df, summary_df, storage_df = benchmark_data_engines(datasets)
    clust_bench, clustered_nhats = benchmark_clustering_mining(datasets.asteroids)
    clf_bench, feat_imp = benchmark_actuarial_classifiers(datasets.launches)
    return BenchmarkSuiteResults(
        engine_workloads=workloads_df,
        engine_summary=summary_df,
        storage_footprint=storage_df,
        clustering_benchmark=clust_bench,
        clustered_nhats=clustered_nhats,
        classification_benchmark=clf_bench,
        feature_importances=feat_imp,
    )
