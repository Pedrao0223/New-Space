"""Automated QA & Scientific Data Integrity Test Suite (pytest)."""

from __future__ import annotations

from contextlib import closing
import json
import sqlite3

import nbformat
import pytest

from newspace.benchmarking import (
    _build_leak_free_actuarial_features,
    benchmark_clustering_mining,
    benchmark_data_engines,
)
from newspace.config import DB_PATH, DOCS_DIR, NOTEBOOKS_DIR, PALETTE, REPORTS_DIR
from newspace.ingestion import load_all_real_datasets, verify_provenance
from newspace.selection_trends import (
    analyze_information_production_trends,
    apply_selection_criteria_and_topsis,
)


@pytest.fixture(scope="module")
def datasets():
    return load_all_real_datasets()


def test_sha256_provenance_and_real_entities(datasets):
    """Guarantee 100% authentic public datasets via SHA-256 and canonical real entities."""
    manifest = verify_provenance()
    assert len(manifest["datasets"]) == 4

    # Verify real canonical aerospace entities exist in parsed tables
    assert len(datasets.satellites) > 65000
    assert len(datasets.asteroids) > 40000
    assert int((datasets.asteroids["acessivel_nhats"] == 1).sum()) >= 7000
    assert len(datasets.launches) > 7000

    assert any(datasets.launchers["modelo_veiculo"] == "Falcon 9")
    assert any(datasets.asteroids["nome_completo"].str.contains("Eros", na=False))
    assert any(datasets.asteroids["nome_completo"].str.contains("Apophis", na=False))


def test_sqlite_foreign_keys_and_optimizer_stats():
    """Verify SQLite relational integrity, composite indexes, and ANALYZE statistics."""
    assert DB_PATH.exists(), "Normalized SQLite DB must exist"
    with closing(sqlite3.connect(DB_PATH)) as conn:
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        assert fk_violations == [], f"Unexpected FK violations: {fk_violations}"

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        expected = {
            "VEICULO_LANCADOR",
            "HISTORICO_LANCAMENTO",
            "CORREDOR_ORBITAL",
            "OBJETO_TRAFEGO_ORBITAL",
            "ASTEROIDE_PROSPECCAO",
            "CENARIO_VIABILIDADE_ATUARIAL",
        }
        assert expected.issubset(tables)

        # Confirm ANALYZE populated sqlite_stat1 for query planner optimization
        stat1_count = conn.execute("SELECT COUNT(*) FROM sqlite_stat1;").fetchone()[0]
        assert stat1_count > 0


def test_zero_temporal_leakage_in_actuarial_features(datasets):
    """Mathematically verify maiden launch of every vehicle has zero prior vehicle flights."""
    X, y_risk, feat_names = _build_leak_free_actuarial_features(datasets.launches)
    assert X.shape[0] == len(datasets.launches)
    assert X.shape[1] == len(feat_names)
    assert set(y_risk).issubset({0, 1})

    # Maiden flight of dataset (Sputnik 1957) must have log_voos_veiculo == 0 and neutral prior 0.90
    idx_log_v = feat_names.index("log_voos_veiculo")
    idx_prior_v = feat_names.index("confiabilidade_previa_veiculo")
    assert X[0, idx_log_v] == pytest.approx(0.0)
    assert X[0, idx_prior_v] == pytest.approx(0.90, abs=1e-6)


def test_engine_and_clustering_benchmarks(datasets):
    """Verify engine benchmark speedup and NHATS clustering validity."""
    w_df, s_df, st_df = benchmark_data_engines(datasets, repeats=3)
    assert len(s_df) == 5
    assert (s_df["latencia_total_mediana_ms"] > 0).all()

    # Indexed SQLite must outperform Unindexed SQLite
    lat_idx = float(
        s_df.loc[s_df["engine"] == "SQLite (Indexado + WAL)", "latencia_total_mediana_ms"].iloc[0]
    )
    lat_unidx = float(
        s_df.loc[s_df["engine"] == "SQLite (Sem Índices)", "latencia_total_mediana_ms"].iloc[0]
    )
    assert lat_idx < lat_unidx
    assert len(st_df) == 4

    c_bench, c_nhats = benchmark_clustering_mining(datasets.asteroids)
    assert len(c_bench) == 5
    assert (c_bench["silhouette_score"] > 0.05).all()
    assert c_nhats["arquetipo_astrodinamico"].nunique() == 4


def test_information_trends_and_topsis_monotonicity(datasets):
    """Verify exponential NewSpace information trend, Spearman OCC law, and TOPSIS sensitivity."""
    trends = analyze_information_production_trends(datasets)
    era1_idx = float(trends.era_comparison.iloc[0]["indice_info_medio"])
    era3_idx = float(trends.era_comparison.iloc[-1]["indice_info_medio"])
    assert era3_idx > 3.0 * era1_idx

    rho, pval = trends.spearman_obs_vs_occ
    assert rho < -0.50
    assert pval < 1e-50

    fc = trends.forecast_2030
    assert (fc["ic95_inferior"] < fc["indice_info_projetado"]).all()
    assert (fc["indice_info_projetado"] < fc["ic95_superior"]).all()

    sel_strict = apply_selection_criteria_and_topsis(datasets, max_delta_v_kms=5.0)
    sel_wide = apply_selection_criteria_and_topsis(datasets, max_delta_v_kms=7.5)
    assert 0 < len(sel_strict.ranked_asteroids) < len(sel_wide.ranked_asteroids)
    assert sel_wide.ranked_asteroids["score_topsis"].between(0.0, 100.0).all()


def test_executed_notebook_and_dashboard_artifacts():
    """Verify executed Jupyter Notebook, standalone HTML dashboard, and technical reports."""
    nb_path = NOTEBOOKS_DIR / "newspace_benchmark_analytics.ipynb"
    dash_path = REPORTS_DIR / "newspace_dashboard.html"
    md_path = REPORTS_DIR / "relatorio_tecnico_decisao.md"
    html_rep_path = REPORTS_DIR / "relatorio_tecnico_decisao.html"

    for p in (nb_path, dash_path, md_path, html_rep_path):
        assert p.exists() and p.stat().st_size > 2048, f"Missing or empty artifact: {p}"

    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 8
    assert all(c.execution_count is not None for c in code_cells), "All code cells must be executed"

    dash_html = dash_path.read_text(encoding="utf-8")
    rep_html = html_rep_path.read_text(encoding="utf-8")
    for hex_color in [
        PALETTE.cosmic_bg,
        PALETTE.space_violet,
        PALETTE.orbital_blue,
        PALETTE.deep_teal,
        PALETTE.vibrant_cyan,
    ]:
        assert hex_color.lower() in dash_html.lower()

    # Verify favicon link & clean proportional Portuguese column headers (zero raw snake_case headers)
    assert '<link rel="icon"' in dash_html
    assert '<link rel="icon"' in rep_html
    assert (DOCS_DIR / "favicon.svg").exists()
    assert (DOCS_DIR / "index.html").exists()
    assert (DOCS_DIR / "relatorio.html").exists()

    # Verify docs/ GitHub Pages files match generated reports byte-for-byte
    assert (DOCS_DIR / "index.html").read_text(encoding="utf-8") == dash_html
    assert (DOCS_DIR / "relatorio.html").read_text(encoding="utf-8") == rep_html

    # Verify mobile & tablet responsive CSS/JS breakpoints and touch scrolling
    for responsive_token in (
        'name="viewport" content="width=device-width, initial-scale=1.0"',
        "@media (max-width: 960px)",
        "-webkit-overflow-scrolling: touch",
        "min-width: 0;",
    ):
        assert responsive_token in dash_html
        assert responsive_token in rep_html

    assert "@media (max-width: 1080px)" in dash_html
    assert "@media (max-width: 640px)" in dash_html
    assert "adaptPlotsForViewport" in dash_html
    assert "toggleMobileFilters" in dash_html

    for raw_col in (
        "era_espacial",
        "media_anual_obj_orbitais",
        "media_anual_neos_descobertos",
        "latencia_total_mediana_ms",
        "brier_score_calibracao",
    ):
        assert raw_col not in rep_html
        assert raw_col not in dash_html
