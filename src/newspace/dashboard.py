"""Customizable Visual Analytics, Notebook ipywidgets Cockpit & Standalone HTML5 Dashboard."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from IPython.display import HTML, clear_output, display
from plotly.subplots import make_subplots

from newspace.benchmarking import BenchmarkSuiteResults
from newspace.config import (
    DEFAULT_CRITERIA,
    DOCS_DIR,
    FAVICON_DATA_URI,
    FAVICON_SVG,
    PALETTE,
    REPORTS_DIR,
)
from newspace.ingestion import MinedDatasets
from newspace.reporting import (
    _render_proportional_html_table,
    format_era_table_for_display,
)
from newspace.selection_trends import (
    SelectionDecisionOutput,
    TrendAnalysisSummary,
    apply_selection_criteria_and_topsis,
)
from newspace.web_assets import (
    CHART_RENDERER_JS,
    DATA_LOADER_JS,
    TABLE_RENDERER_JS,
    TOPSIS_ENGINE_JS,
    UI_CONTROLLER_JS,
    build_dashboard_css,
    figure_to_pure_dict,
)


FONT_STACK = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


SHORT_STORAGE_LABELS = {
    "Apache Parquet Colunar (Compressão ZSTD/Snappy)": "Parquet Colunar (ZSTD)",
    "SQLite Relacional Normalizado (Sem Índices)": "SQLite (Sem Índices)",
    "Arquivos Brutos Originais (JSON + CSV + TSV)": "Arquivos Brutos (JSON/CSV)",
    "SQLite Relacional Normalizado + Índices B-Tree": "SQLite + Índices B-Tree",
}

SHORT_MODEL_LABELS = {
    "Regressão Logística L2 (Baseline Atuarial)": "Reg. Logística L2",
    "Extra Trees (200 Árvores)": "Extra Trees",
    "Rede Neural MLP (64x32 ReLU)": "Rede Neural MLP",
    "Random Forest (200 Árvores)": "Random Forest",
    "HistGradientBoosting (GBDT)": "HistGradientBoosting",
}

SHORT_FACTOR_LABELS = {
    "Taxa Histórica Prévia do Foguete (Bayes)": "Histórico Prévio do Foguete",
    "Histórico Prévio da Agência/Operador": "Histórico Prévio do Operador",
    "Era Tecnológica (Ano do Lançamento)": "Era Tecnológica (Ano)",
    "Maturidade da Base de Lançamento": "Maturidade da Base",
    "Experiência Acumulada da Agência/Operador": "Experiência do Operador",
    "Herança de Voo Acumulada do Foguete": "Herança de Voos do Foguete",
    "Idade Operacional do Modelo (Anos)": "Idade Operacional (Anos)",
    "Complexidade de Dispensação (Nº de Cargas)": "Complexidade de Cargas",
    "Regime Energético (Escape / Espaço Profundo)": "Espaço Profundo (Escape)",
}

SHORT_CORRIDOR_LABELS = {
    "LEO_VLEO": "LEO Muito Baixa",
    "LEO_CONSTELACAO": "LEO Constelações",
    "LEO_SSO_ALTA": "LEO Alta / SSO",
    "MEO_NAVEGACAO": "MEO Navegação",
    "GEO_CINTURAO": "GEO Cinturão",
    "HEO_CISLUNAR": "HEO / Cislunar",
    "ESPACO_PROFUNDO": "Espaço Profundo",
}


def _apply_cosmic_layout(
    fig: go.Figure,
    title: str,
    height: int = 460,
    top_legend: bool = True,
) -> go.Figure:
    """Apply balanced margins, auto-margin axes, and the minimalist deep-space palette."""
    legend_cfg = (
        dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0, 0, 0, 0)",
            borderwidth=0,
            font=dict(size=11.5, color=PALETTE.text_muted),
        )
        if top_legend
        else dict(
            bgcolor="rgba(0, 0, 0, 0)",
            borderwidth=0,
            font=dict(size=11.5, color=PALETTE.text_muted),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color=PALETTE.text_primary, family=FONT_STACK),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor=PALETTE.cosmic_bg,
        plot_bgcolor=PALETTE.cosmic_bg,
        font=dict(color=PALETTE.text_primary, family=FONT_STACK, size=12),
        height=height,
        margin=dict(l=24, r=28, t=76, b=44),
        legend=legend_cfg,
        hoverlabel=dict(
            bgcolor="#0B0F1C",
            bordercolor="rgba(148, 163, 184, 0.25)",
            font=dict(color=PALETTE.text_primary, size=12, family=FONT_STACK),
        ),
    )
    fig.update_xaxes(
        gridcolor=PALETTE.grid_line,
        zerolinecolor=PALETTE.grid_line,
        linecolor=PALETTE.grid_line,
        tickfont=dict(color=PALETTE.text_muted),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=PALETTE.grid_line,
        zerolinecolor=PALETTE.grid_line,
        linecolor=PALETTE.grid_line,
        tickfont=dict(color=PALETTE.text_muted),
        automargin=True,
    )
    return fig


def fig_engine_latency_card(bench: BenchmarkSuiteResults) -> go.Figure:
    """Standalone crisp horizontal bar chart for Engine Latency & Speedup."""
    eng = bench.engine_summary.sort_values("latencia_total_mediana_ms", ascending=False)
    max_x = float(eng["latencia_total_mediana_ms"].max())
    bar_colors = [
        PALETTE.vibrant_cyan
        if sp == eng["speedup_vs_sqlite_sem_indice"].max()
        else PALETTE.deep_teal
        if sp >= 2.0
        else PALETTE.orbital_blue
        for sp in eng["speedup_vs_sqlite_sem_indice"]
    ]

    fig = go.Figure(
        go.Bar(
            y=eng["engine"],
            x=eng["latencia_total_mediana_ms"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[
                f"  {ms:.1f} ms ({sp:.1f}x)"
                for ms, sp in zip(
                    eng["latencia_total_mediana_ms"], eng["speedup_vs_sqlite_sem_indice"]
                )
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Latência Mediana Total: %{x:.2f} ms<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, max_x * 1.32], title_text="Latência Total nas 3 Cargas (ms — Menor é Melhor)")
    fig.update_layout(showlegend=False)
    return _apply_cosmic_layout(
        fig, "Desempenho de Consulta: Motores Analíticos & Relacionais", height=400
    )


def fig_storage_footprint_card(bench: BenchmarkSuiteResults) -> go.Figure:
    """Standalone crisp horizontal bar chart for Disk Footprint & Compression."""
    st = bench.storage_footprint.sort_values("tamanho_mb", ascending=False).copy()
    st["short_label"] = st["formato_persistencia"].map(
        lambda s: SHORT_STORAGE_LABELS.get(s, s)
    )
    max_mb = float(st["tamanho_mb"].max())
    colors = [
        PALETTE.vibrant_cyan if "Parquet" in lbl else PALETTE.deep_teal if "Sem" in lbl else PALETTE.orbital_blue
        for lbl in st["short_label"]
    ]

    fig = go.Figure(
        go.Bar(
            y=st["short_label"],
            x=st["tamanho_mb"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[
                f"  {mb:.2f} MB ({cr:.2f}x)"
                for mb, cr in zip(st["tamanho_mb"], st["razao_compressao_vs_bruto"])
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Espaço em Disco: %{x:.2f} MB<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, max_mb * 1.34], title_text="Tamanho em Disco (MB — 122.200 Registros Reais)")
    fig.update_layout(showlegend=False)
    return _apply_cosmic_layout(
        fig, "Eficiência de Armazenamento & Compressão Física", height=400
    )


def fig_engine_benchmark(bench: BenchmarkSuiteResults) -> go.Figure:
    """Overlap-free 2-row figure for Jupyter Notebook displaying Latency and Storage."""
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.22,
        subplot_titles=(
            "Latência Mediana Total nas 3 Cargas Analíticas (ms — Menor é Melhor)",
            "Pegada de Armazenamento em Disco (MB — 122.200 Registros Reais)",
        ),
    )

    eng = bench.engine_summary.sort_values("latencia_total_mediana_ms", ascending=False)
    max_lat = float(eng["latencia_total_mediana_ms"].max())
    fig.add_trace(
        go.Bar(
            y=eng["engine"],
            x=eng["latencia_total_mediana_ms"],
            orientation="h",
            marker=dict(color=PALETTE.vibrant_cyan, line=dict(width=0)),
            text=[
                f"  {ms:.1f} ms ({sp:.1f}x)"
                for ms, sp in zip(eng["latencia_total_mediana_ms"], eng["speedup_vs_sqlite_sem_indice"])
            ],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )

    st = bench.storage_footprint.sort_values("tamanho_mb", ascending=False).copy()
    st["short_label"] = st["formato_persistencia"].map(lambda s: SHORT_STORAGE_LABELS.get(s, s))
    max_mb = float(st["tamanho_mb"].max())
    fig.add_trace(
        go.Bar(
            y=st["short_label"],
            x=st["tamanho_mb"],
            orientation="h",
            marker=dict(color=PALETTE.deep_teal, line=dict(width=0)),
            text=[
                f"  {mb:.2f} MB ({cr:.2f}x)"
                for mb, cr in zip(st["tamanho_mb"], st["razao_compressao_vs_bruto"])
            ],
            textposition="outside",
            cliponaxis=False,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(range=[0, max_lat * 1.28], row=1, col=1)
    fig.update_xaxes(range=[0, max_mb * 1.28], row=2, col=1)
    fig.update_layout(showlegend=False)
    return _apply_cosmic_layout(
        fig,
        "Benchmarking de Engenharia de Dados: Motores Analíticos & Persistência Relacional",
        height=640,
    )


def fig_ml_classifiers_card(bench: BenchmarkSuiteResults) -> go.Figure:
    """Standalone crisp grouped column chart for Supervised Actuarial Risk Models."""
    clf = bench.classification_benchmark.copy()
    clf["short_model"] = clf["modelo_atuarial"].map(lambda s: SHORT_MODEL_LABELS.get(s, s))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="ROC-AUC (Discriminação)",
            x=clf["short_model"],
            y=clf["roc_auc"],
            marker_color=PALETTE.vibrant_cyan,
            text=[f"{v:.3f}" for v in clf["roc_auc"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            name="1 - Brier (Calibração Atuarial)",
            x=clf["short_model"],
            y=1.0 - clf["brier_score_calibracao"],
            marker_color=PALETTE.orbital_blue,
            text=[f"{1.0 - v:.3f}" for v in clf["brier_score_calibracao"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_yaxes(range=[0.65, 1.03], title_text="Score de Validação Cruzada (5-Fold CV)")
    fig.update_xaxes(tickangle=0)
    fig.update_layout(barmode="group")
    return _apply_cosmic_layout(
        fig, "Modelos Preditivos de Sinistro Atuarial (7.418 Voos)", height=430, top_legend=True
    )


def fig_risk_factors_card(bench: BenchmarkSuiteResults) -> go.Figure:
    """Standalone crisp horizontal bar chart for Empirical Launch Risk Drivers."""
    imp = bench.feature_importances.sort_values("importancia_relativa_pct", ascending=True).copy()
    imp["short_factor"] = imp["fator_risco"].map(lambda s: SHORT_FACTOR_LABELS.get(s, s))
    max_imp = float(imp["importancia_relativa_pct"].max())

    fig = go.Figure(
        go.Bar(
            y=imp["short_factor"],
            x=imp["importancia_relativa_pct"],
            orientation="h",
            marker=dict(color=PALETTE.deep_teal, line=dict(width=0)),
            text=[f"  {v:.1f}%" for v in imp["importancia_relativa_pct"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Importância Relativa: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, max_imp * 1.25], title_text="Importância Relativa Explicativa (%)")
    fig.update_layout(showlegend=False)
    return _apply_cosmic_layout(
        fig, "Importância Empírica dos Fatores de Risco de Lançamento", height=430
    )


def fig_ml_benchmark(bench: BenchmarkSuiteResults) -> go.Figure:
    """Overlap-free 2-row figure for Jupyter Notebook combining Classifiers & Risk Drivers."""
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.20,
        subplot_titles=(
            "Modelos Atuariais de Sinistro (5-Fold CV em 7.418 Lançamentos)",
            "Importância Empírica dos Fatores de Risco de Lançamento (%)",
        ),
    )

    clf = bench.classification_benchmark.copy()
    clf["short_model"] = clf["modelo_atuarial"].map(lambda s: SHORT_MODEL_LABELS.get(s, s))

    fig.add_trace(
        go.Bar(
            name="ROC-AUC (Discriminação)",
            x=clf["short_model"],
            y=clf["roc_auc"],
            marker_color=PALETTE.vibrant_cyan,
            text=[f"{v:.3f}" for v in clf["roc_auc"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="1 - Brier (Calibração Atuarial)",
            x=clf["short_model"],
            y=1.0 - clf["brier_score_calibracao"],
            marker_color=PALETTE.orbital_blue,
            text=[f"{1.0 - v:.3f}" for v in clf["brier_score_calibracao"]],
            textposition="outside",
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )

    imp = bench.feature_importances.sort_values("importancia_relativa_pct", ascending=True).copy()
    imp["short_factor"] = imp["fator_risco"].map(lambda s: SHORT_FACTOR_LABELS.get(s, s))
    max_imp = float(imp["importancia_relativa_pct"].max())

    fig.add_trace(
        go.Bar(
            y=imp["short_factor"],
            x=imp["importancia_relativa_pct"],
            orientation="h",
            marker_color=PALETTE.deep_teal,
            text=[f"  {v:.1f}%" for v in imp["importancia_relativa_pct"]],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(range=[0.65, 1.03], row=1, col=1)
    fig.update_xaxes(tickangle=0, row=1, col=1)
    fig.update_xaxes(range=[0, max_imp * 1.22], row=2, col=1)
    fig.update_layout(barmode="group")
    return _apply_cosmic_layout(
        fig,
        "Benchmarking de Mineração Preditiva: Subscrição Atuarial Aeroespacial",
        height=720,
        top_legend=True,
    )


def fig_information_trends(trends: TrendAnalysisSummary) -> go.Figure:
    """3-panel historical & projected time-series of Aerospace Information Production (1957-2030)."""
    hist = trends.annual_series[trends.annual_series["ano"] <= 2025].copy()
    fc = trends.forecast_2030.copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "1. Produção Anual de Telemetria & Objetos em Órbita (NORAD SATCAT, 1957–2025)",
            "2. Descobertas Anuais de Asteroides Próximos à Terra (NASA JPL SBDB)",
            "3. Índice Composto de Produção de Informações (Base 2010 = 100) + Projeção 2030 (IC 95%)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=hist["ano"],
            y=hist["cargas_uteis_pay_ano"],
            mode="lines",
            name="Satélites / Cargas Úteis (PAY)",
            line=dict(color=PALETTE.vibrant_cyan, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.14)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hist["ano"],
            y=hist["detritos_estagios_ano"],
            mode="lines",
            name="Detritos & Estágios (DEB + R/B)",
            line=dict(color=PALETTE.accent_gold, width=1.8, dash="dot"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=hist["ano"],
            y=hist["neos_descobertos_ano"],
            name="NEOs Descobertos / Ano",
            marker_color=PALETTE.deep_teal,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=hist["ano"],
            y=hist["indice_producao_informacao"],
            mode="lines+markers",
            marker=dict(size=4),
            name="Índice Observado (1957–2025)",
            line=dict(color=PALETTE.vibrant_cyan, width=2.8),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc["ano"]) + list(fc["ano"])[::-1],
            y=list(fc["ic95_superior"]) + list(fc["ic95_inferior"])[::-1],
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            name="IC 95% (2026–2030)",
            hoverinfo="skip",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[int(hist["ano"].iloc[-1])] + list(fc["ano"]),
            y=[float(hist["indice_producao_informacao"].iloc[-1])]
            + list(fc["indice_info_projetado"]),
            mode="lines+markers",
            name="Projeção Log-Linear (2026–2030)",
            line=dict(color=PALETTE.accent_mint, width=2.6, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        ),
        row=3,
        col=1,
    )

    for r in (1, 2, 3):
        fig.add_vline(x=2015, line_dash="dash", line_color="#A78BFA", line_width=1.4, row=r, col=1)

    fig.add_annotation(
        x=2015,
        y=float(hist["cargas_uteis_pay_ano"].max()) * 0.86,
        text="Inflexão NewSpace (2015)<br>Reusabilidade & Megaconstelações",
        showarrow=True,
        arrowhead=2,
        ax=-90,
        ay=-15,
        font=dict(color="#E9D5FF", size=11),
        bgcolor="rgba(11, 15, 28, 0.92)",
        bordercolor="#A78BFA",
        row=1,
        col=1,
    )

    out = _apply_cosmic_layout(
        fig,
        "Tendência Empírica de Produção de Informações Espaciais e Projeção 2030",
        height=790,
        top_legend=True,
    )
    out.update_layout(margin=dict(l=40, r=28, t=105, b=45))
    return out


def fig_observability_card(trends: TrendAnalysisSummary) -> go.Figure:
    """Standalone card: Empirical Law of Information Volume vs NASA Orbit Uncertainty (OCC)."""
    occ = trends.observability_vs_uncertainty
    max_y = float(occ["mediana_observacoes"].max())
    fig = go.Figure(
        go.Bar(
            x=[f"OCC {int(c)}" for c in occ["codigo_incerteza_occ_int"]],
            y=occ["mediana_observacoes"],
            marker=dict(
                color=occ["codigo_incerteza_occ_int"],
                colorscale=[
                    [0.0, PALETTE.vibrant_cyan],
                    [0.5, PALETTE.orbital_blue],
                    [1.0, PALETTE.accent_coral],
                ],
            ),
            text=[f"{int(v)}" for v in occ["mediana_observacoes"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>Mediana de Observações: %{y}<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, max_y * 1.18], title_text="Mediana de Observações Astrométricas")
    fig.update_xaxes(title_text="Código de Incerteza Orbital NASA (0 = Órbita Certa, 9 = Incerta)")
    fig.update_layout(showlegend=False)
    return _apply_cosmic_layout(
        fig, "Lei Informação × Risco: Observações por Nível de Incerteza OCC", height=430
    )


def fig_corridors_card(datasets: MinedDatasets) -> go.Figure:
    """Standalone card: Active Satellites vs Debris across the 7 physical Orbital Corridors."""
    cor = datasets.corridors.copy()
    cor["short_corridor"] = cor["id_corredor"].map(lambda c: SHORT_CORRIDOR_LABELS.get(c, c))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cor["short_corridor"],
            y=cor["satelites_ativos"],
            name="Satélites Ativos Manobráveis",
            marker_color=PALETTE.vibrant_cyan,
        )
    )
    fig.add_trace(
        go.Bar(
            x=cor["short_corridor"],
            y=cor["detritos_e_estagios_orbita"],
            name="Detritos & Estágios Inertes (DEB + R/B)",
            marker_color=PALETTE.accent_coral,
        )
    )
    fig.update_xaxes(tickangle=-18)
    fig.update_yaxes(title_text="Objetos Atualmente em Órbita (NORAD)")
    fig.update_layout(barmode="stack")
    return _apply_cosmic_layout(
        fig, "Bifurcação de Tráfego nos Corredores Orbitais", height=430, top_legend=True
    )


def fig_observability_and_corridors(
    datasets: MinedDatasets, trends: TrendAnalysisSummary
) -> go.Figure:
    """Overlap-free 2-row figure for Jupyter Notebook showing OCC Law & Corridor Congestion."""
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.22,
        subplot_titles=(
            "Lei Informação × Risco: Mediana de Observações por Incerteza OCC (NASA JPL)",
            "Tráfego Real nos Corredores Orbitais: Satélites Ativos vs Detritos Inertes",
        ),
    )

    occ = trends.observability_vs_uncertainty
    max_y = float(occ["mediana_observacoes"].max())
    fig.add_trace(
        go.Bar(
            x=[f"OCC {int(c)}" for c in occ["codigo_incerteza_occ_int"]],
            y=occ["mediana_observacoes"],
            marker=dict(
                color=occ["codigo_incerteza_occ_int"],
                colorscale=[
                    [0.0, PALETTE.vibrant_cyan],
                    [0.5, PALETTE.orbital_blue],
                    [1.0, PALETTE.accent_coral],
                ],
            ),
            text=[f"{int(v)}" for v in occ["mediana_observacoes"]],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    cor = datasets.corridors.copy()
    cor["short_corridor"] = cor["id_corredor"].map(lambda c: SHORT_CORRIDOR_LABELS.get(c, c))
    fig.add_trace(
        go.Bar(
            x=cor["short_corridor"],
            y=cor["satelites_ativos"],
            name="Satélites Ativos Manobráveis",
            marker_color=PALETTE.vibrant_cyan,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=cor["short_corridor"],
            y=cor["detritos_e_estagios_orbita"],
            name="Detritos & Estágios Inertes (DEB + R/B)",
            marker_color=PALETTE.accent_coral,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(range=[0, max_y * 1.18], row=1, col=1)
    fig.update_xaxes(tickangle=0, row=2, col=1)
    fig.update_layout(barmode="stack")
    return _apply_cosmic_layout(
        fig,
        "Qualidade Informacional Astrométrica & Bifurcação nos Corredores Orbitais",
        height=700,
        top_legend=True,
    )


def fig_topsis_prospecting_map(selection: SelectionDecisionOutput) -> go.Figure:
    """Astrodynamic Trade-Off Scatter of qualified NASA NHATS targets colored by TOPSIS score."""
    cand = selection.ranked_asteroids.head(250).copy()
    if cand.empty:
        return _apply_cosmic_layout(go.Figure(), "Nenhum Asteroide Atendeu aos Filtros Selecionados", 420)

    sizes = np.clip(np.log1p(cand["diametro_fisico_ref_m"]) * 4.2, 8.0, 32.0)
    spec_clean = cand["classe_espectral"].replace("NAO_ESPECTROGRAFADO", "Não Medido")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cand["min_delta_v_kms"],
            y=cand["duracao_min_dv_dias"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=cand["score_topsis"],
                colorscale=[
                    [0.0, PALETTE.space_violet],
                    [0.35, PALETTE.orbital_blue],
                    [0.70, PALETTE.deep_teal],
                    [1.0, PALETTE.vibrant_cyan],
                ],
                colorbar=dict(title="Score<br>TOPSIS"),
                line=dict(width=0.6, color="rgba(226,232,240,0.35)"),
                opacity=0.88,
            ),
            customdata=np.column_stack(
                [
                    cand["ranking_topsis"],
                    cand["nome_completo"],
                    spec_clean,
                    np.round(cand["diametro_fisico_ref_m"], 1),
                    cand["num_observacoes"],
                    cand["codigo_incerteza_occ"],
                    cand["score_topsis"],
                ]
            ),
            hovertemplate=(
                "<b>#%{customdata[0]} — %{customdata[1]}</b><br>"
                "Δv Mínimo: %{x:.3f} km/s | Duração: %{y:.0f} dias<br>"
                "Espectro: %{customdata[2]} | Diâmetro Ref.: %{customdata[3]} m<br>"
                "Observações NASA: %{customdata[4]} (OCC %{customdata[5]})<br>"
                "<b>Score TOPSIS: %{customdata[6]:.2f}</b><extra></extra>"
            ),
            name="Candidatos Qualificados",
        )
    )

    top6 = cand.head(6)
    fig.add_trace(
        go.Scatter(
            x=top6["min_delta_v_kms"],
            y=top6["duracao_min_dv_dias"],
            mode="text",
            text=[f"#{r} {p}" for r, p in zip(top6["ranking_topsis"], top6["pdes"])],
            textposition="top center",
            textfont=dict(color="#FFFFFF", size=11),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_xaxes(title_text="Δv Mínimo de Missão NASA NHATS (km/s — Menor = Mais Eficiente)")
    fig.update_yaxes(title_text="Duração da Missão de Menor Δv (Dias)")
    fig.update_layout(showlegend=False)
    return _apply_cosmic_layout(
        fig,
        f"Mapa de Decisão Multicritério TOPSIS — Top {len(cand)} Alvos Qualificados (NASA JPL NHATS)",
        height=490,
    )


def fig_actuarial_scenarios_matrix(selection: SelectionDecisionOutput) -> go.Figure:
    """Visual underwriting matrix mapping TOPSIS Opportunity Score vs Combined Actuarial Risk Score."""
    scen = selection.actuarial_scenarios.copy()
    if scen.empty:
        return _apply_cosmic_layout(go.Figure(), "Sem Cenários Atuariais para os Filtros Atuais", 420)

    rating_colors = {
        "AA": PALETTE.accent_mint,
        "A": PALETTE.vibrant_cyan,
        "BB": PALETTE.accent_gold,
        "B": "#FB923C",
        "C": PALETTE.accent_coral,
    }

    fig = go.Figure()
    for rating in ["AA", "A", "BB", "B", "C"]:
        sub = scen[scen["rating_atuarial"] == rating]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["score_risco_combinado"],
                y=sub["score_topsis"],
                mode="markers",
                name=f"Rating {rating}",
                marker=dict(
                    size=13,
                    color=rating_colors[rating],
                    symbol="circle",
                    line=dict(color="#FFFFFF", width=1),
                ),
                customdata=np.column_stack(
                    [
                        sub["id_cenario"],
                        sub["nome_asteroide"],
                        sub["modelo_veiculo"],
                        sub["min_delta_v_kms"],
                        sub["confiabilidade_wilson_pct"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (Rating " + rating + ")<br>"
                    "Alvo: %{customdata[1]}<br>"
                    "Foguete: %{customdata[2]} (Wilson 95%: %{customdata[4]:.2f}%)<br>"
                    "Δv: %{customdata[3]:.3f} km/s<br>"
                    "Risco Combinado: %{x:.2f} | Score TOPSIS: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_xaxes(title_text="Score de Risco Atuarial Combinado (Menor = Mais Seguro)")
    fig.update_yaxes(title_text="Score de Atratividade Multicritério TOPSIS (Maior = Melhor)")
    return _apply_cosmic_layout(
        fig,
        "Matriz Integrada de Subscrição Atuarial: Retorno Técnico TOPSIS vs Risco Operacional",
        height=470,
        top_legend=True,
    )


def create_notebook_interactive_cockpit(datasets: MinedDatasets) -> widgets.VBox:
    """Create live customizable ipywidgets dashboard inside Jupyter Notebook."""
    style = {"description_width": "185px"}
    layout_sl = widgets.Layout(width="460px")

    sl_dv = widgets.FloatSlider(
        value=DEFAULT_CRITERIA.max_delta_v_kms,
        min=4.0,
        max=11.0,
        step=0.25,
        description="Δv Máximo (km/s):",
        style=style,
        layout=layout_sl,
    )
    sl_dur = widgets.IntSlider(
        value=DEFAULT_CRITERIA.max_duration_days,
        min=120,
        max=450,
        step=30,
        description="Duração Máx (dias):",
        style=style,
        layout=layout_sl,
    )
    sl_occ = widgets.IntSlider(
        value=DEFAULT_CRITERIA.max_orbit_uncertainty,
        min=0,
        max=8,
        step=1,
        description="Incerteza Orbital (OCC ≤):",
        style=style,
        layout=layout_sl,
    )
    sl_obs = widgets.IntSlider(
        value=DEFAULT_CRITERIA.min_observations,
        min=10,
        max=500,
        step=10,
        description="Observações Mínimas:",
        style=style,
        layout=layout_sl,
    )
    sl_diam = widgets.FloatSlider(
        value=25.0,
        min=5.0,
        max=500.0,
        step=15.0,
        description="Diâmetro Mín (m):",
        style=style,
        layout=layout_sl,
    )
    chk_spec = widgets.Checkbox(
        value=False,
        description="Exigir Espectroscopia Mineralógica Já Confirmada pela NASA",
        indent=False,
    )

    w_astro = widgets.FloatSlider(
        value=0.40, min=0.05, max=1.0, step=0.05, description="Peso Eficiência Δv:", style=style, layout=layout_sl
    )
    w_info = widgets.FloatSlider(
        value=0.25, min=0.05, max=1.0, step=0.05, description="Peso Maturidade Info:", style=style, layout=layout_sl
    )
    w_scale = widgets.FloatSlider(
        value=0.20, min=0.05, max=1.0, step=0.05, description="Peso Porte Físico:", style=style, layout=layout_sl
    )
    w_traj = widgets.FloatSlider(
        value=0.15, min=0.05, max=1.0, step=0.05, description="Peso Janelas Viáveis:", style=style, layout=layout_sl
    )

    out_area = widgets.Output()

    def _refresh(_change=None):
        res = apply_selection_criteria_and_topsis(
            datasets,
            max_delta_v_kms=float(sl_dv.value),
            max_duration_days=int(sl_dur.value),
            max_orbit_uncertainty=int(sl_occ.value),
            min_observations=int(sl_obs.value),
            min_diameter_m=float(sl_diam.value),
            only_spectroscopically_characterized=bool(chk_spec.value),
            weights=(
                float(w_astro.value),
                float(w_info.value),
                float(w_scale.value),
                float(w_traj.value),
            ),
        )
        with out_area:
            clear_output(wait=True)
            n_q = len(res.ranked_asteroids)
            best_name = res.ranked_asteroids["nome_completo"].iloc[0] if n_q > 0 else "N/A"
            best_dv = f"{res.ranked_asteroids['min_delta_v_kms'].iloc[0]:.3f} km/s" if n_q > 0 else "-"
            aa_count = (
                int(res.actuarial_scenarios["rating_atuarial"].isin(["AA", "A"]).sum())
                if not res.actuarial_scenarios.empty
                else 0
            )
            kpi_html = f"""
            <div style="background:{PALETTE.cosmic_bg};border:1px solid {PALETTE.vibrant_cyan};border-radius:10px;padding:14px 18px;color:{PALETTE.text_primary};font-family:Segoe UI,sans-serif;display:flex;gap:24px;flex-wrap:wrap;margin-bottom:10px;">
              <div><div style="font-size:11px;color:{PALETTE.text_muted};">ALVOS QUALIFICADOS</div><div style="font-size:22px;font-weight:700;color:{PALETTE.vibrant_cyan};">{n_q:,}</div></div>
              <div><div style="font-size:11px;color:{PALETTE.text_muted};">ALVO #1 TOPSIS</div><div style="font-size:18px;font-weight:700;color:#fff;">{best_name}</div></div>
              <div><div style="font-size:11px;color:{PALETTE.text_muted};">Δv DO ALVO #1</div><div style="font-size:22px;font-weight:700;color:{PALETTE.accent_mint};">{best_dv}</div></div>
              <div><div style="font-size:11px;color:{PALETTE.text_muted};">CENÁRIOS ATUARIAIS AA/A</div><div style="font-size:22px;font-weight:700;color:{PALETTE.vibrant_cyan};">{aa_count}</div></div>
            </div>
            """
            display(HTML(kpi_html))
            fig_topsis_prospecting_map(res).show()
            if n_q > 0:
                top_show = res.ranked_asteroids.head(12).copy()
                disp_tbl = pd.DataFrame(
                    {
                        "Rank": top_show["ranking_topsis"],
                        "Designação": top_show["pdes"],
                        "Asteroide (NASA JPL)": top_show["nome_completo"],
                        "Espectro": top_show["classe_espectral"].replace("NAO_ESPECTROGRAFADO", "Não Medido"),
                        "Δv Mín. (km/s)": top_show["min_delta_v_kms"],
                        "Duração (d)": top_show["duracao_min_dv_dias"].astype(int),
                        "Diâm. Ref. (m)": top_show["diametro_fisico_ref_m"].round(1),
                        "Obs.": top_show["num_observacoes"].astype(int),
                        "OCC": top_show["codigo_incerteza_occ"].astype(int),
                        "Score TOPSIS": top_show["score_topsis"],
                    }
                )
                display(disp_tbl)

    for ctrl in [sl_dv, sl_dur, sl_occ, sl_obs, sl_diam, chk_spec, w_astro, w_info, w_scale, w_traj]:
        ctrl.observe(_refresh, names="value")

    header = widgets.HTML(
        f"<div style='background:linear-gradient(90deg,{PALETTE.space_violet},{PALETTE.orbital_blue});"
        f"padding:12px 18px;border-radius:8px;color:#fff;font-family:Segoe UI,sans-serif;margin-bottom:8px;'>"
        f"<b>Painel Customizável NewSpace (MCDA-TOPSIS &amp; Risco Atuarial em Tempo Real)</b><br>"
        f"<span style='font-size:12px;color:#cbd5e1;'>Ajuste os critérios físicos, observacionais e pesos decisórios abaixo:</span></div>"
    )
    grid_controls = widgets.HBox(
        [
            widgets.VBox([sl_dv, sl_dur, sl_occ, sl_obs, sl_diam, chk_spec]),
            widgets.VBox([w_astro, w_info, w_scale, w_traj]),
        ]
    )
    _refresh()
    return widgets.VBox([header, grid_controls, out_area])


def _write_modular_dashboard_bundle(
    target_dir: Path,
    html_filename: str,
    html_content: str,
    css_content: str,
    data_js_content: str,
    plotly_bundle_src: Path,
) -> Path:
    """Write modular HTML, CSS, local Plotly vendor bundle, data module, and JS modules."""
    target_dir.mkdir(parents=True, exist_ok=True)
    css_dir = target_dir / "assets" / "css"
    js_dir = target_dir / "assets" / "js"
    data_dir = target_dir / "assets" / "data"
    vendor_dir = target_dir / "assets" / "vendor"

    for folder in (css_dir, js_dir, data_dir, vendor_dir):
        folder.mkdir(parents=True, exist_ok=True)

    (target_dir / "favicon.svg").write_text(FAVICON_SVG, encoding="utf-8")
    (css_dir / "dashboard.css").write_text(css_content, encoding="utf-8")
    (data_dir / "dashboard-data.js").write_text(data_js_content, encoding="utf-8")
    (js_dir / "data-loader.js").write_text(DATA_LOADER_JS, encoding="utf-8")
    (js_dir / "topsis-engine.js").write_text(TOPSIS_ENGINE_JS, encoding="utf-8")
    (js_dir / "table-renderer.js").write_text(TABLE_RENDERER_JS, encoding="utf-8")
    (js_dir / "chart-renderer.js").write_text(CHART_RENDERER_JS, encoding="utf-8")
    (js_dir / "ui-controller.js").write_text(UI_CONTROLLER_JS, encoding="utf-8")

    vendor_dst = vendor_dir / "plotly.min.js"
    if plotly_bundle_src.exists():
        shutil.copyfile(plotly_bundle_src, vendor_dst)

    html_path = target_dir / html_filename
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def export_standalone_custom_dashboard(
    datasets: MinedDatasets,
    bench: BenchmarkSuiteResults,
    trends: TrendAnalysisSummary,
    selection: SelectionDecisionOutput,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Export modular, responsive HTML5 cockpit with pure JSON data + sync to docs/."""
    # Convert all 8 static analytical figures into pure JSON dicts (zero binary bdata!)
    figures_payload = {
        "engLat": figure_to_pure_dict(fig_engine_latency_card(bench)),
        "engSto": figure_to_pure_dict(fig_storage_footprint_card(bench)),
        "mlClf": figure_to_pure_dict(fig_ml_classifiers_card(bench)),
        "mlImp": figure_to_pure_dict(fig_risk_factors_card(bench)),
        "trends": figure_to_pure_dict(fig_information_trends(trends)),
        "obsLaw": figure_to_pure_dict(fig_observability_card(trends)),
        "corridors": figure_to_pure_dict(fig_corridors_card(datasets)),
        "actuarialMat": figure_to_pure_dict(fig_actuarial_scenarios_matrix(selection)),
    }

    # Formatted proportional Era Comparison HTML table for Tab 2
    era_html_table = _render_proportional_html_table(
        format_era_table_for_display(trends.era_comparison), right_align_from_col=1
    )

    # Formatted Actuarial Scenarios HTML table for Tab 4
    raw_scen = selection.actuarial_scenarios.head(12)
    scen_disp = pd.DataFrame(
        {
            "Cenário": raw_scen["id_cenario"],
            "Asteroide Alvo": raw_scen["nome_asteroide"],
            "Veículo Lançador": raw_scen["modelo_veiculo"],
            "Corredor": raw_scen["id_corredor_transito"].map(
                lambda c: SHORT_CORRIDOR_LABELS.get(c, c)
            ),
            "Δv (km/s)": raw_scen["min_delta_v_kms"].map(lambda v: f"{v:.3f}"),
            "Wilson 95%": raw_scen["confiabilidade_wilson_pct"].map(lambda v: f"{v:.2f}%"),
            "Score TOPSIS": raw_scen["score_topsis"].map(lambda v: f"{v:.2f}"),
            "Risco Combinado": raw_scen["score_risco_combinado"].map(lambda v: f"{v:.2f}"),
            "Rating": raw_scen["rating_atuarial"],
        }
    )
    scen_html_table = _render_proportional_html_table(scen_disp, right_align_from_col=4)

    # Prepare compact JSON of all 7,169 real NASA NHATS targets for instant client-side TOPSIS customization
    nhats_all = datasets.asteroids[datasets.asteroids["acessivel_nhats"] == 1].copy()
    nhats_all = nhats_all.dropna(
        subset=["min_delta_v_kms", "duracao_min_dv_dias", "diametro_fisico_ref_m"]
    ).copy()

    client_records = []
    for _, r in nhats_all.iterrows():
        raw_sp = str(r["classe_espectral"])
        client_records.append(
            {
                "spkid": int(r["spkid"]),
                "pdes": str(r["pdes"]),
                "name": str(r["nome_completo"]),
                "spec": "Não Medido" if raw_sp == "NAO_ESPECTROGRAFADO" else raw_sp,
                "has_spec": 0 if raw_sp == "NAO_ESPECTROGRAFADO" else 1,
                "grp": str(r["grupo_mineralogico"]),
                "dv": round(float(r["min_delta_v_kms"]), 3),
                "dur": int(r["duracao_min_dv_dias"]),
                "diam": round(float(r["diametro_fisico_ref_m"]), 1),
                "diam_m_real": 1 if pd.notna(r["diametro_medido_km"]) else 0,
                "obs": int(r["num_observacoes"]) if pd.notna(r["num_observacoes"]) else 0,
                "arc": int(r["arco_observacao_dias"]) if pd.notna(r["arco_observacao_dias"]) else 0,
                "occ": int(r["codigo_incerteza_occ"]) if pd.notna(r["codigo_incerteza_occ"]) else 9,
                "traj": int(r["trajetorias_viaveis"]) if pd.notna(r["trajetorias_viaveis"]) else 1,
            }
        )

    total_records = (
        len(datasets.satellites)
        + len(datasets.asteroids)
        + len(datasets.launches)
        + len(datasets.launchers)
        + len(datasets.corridors)
    )
    best_eng = bench.engine_summary.iloc[0]
    fc_2030 = trends.forecast_2030[trends.forecast_2030["ano"] == 2030].iloc[0]

    dashboard_payload = {
        "meta": {
            "totalAuditedRecords": int(total_records),
            "nhatsAccessibleCount": len(client_records),
        },
        "nhatsRecords": client_records,
        "figures": figures_payload,
    }
    payload_json = json.dumps(dashboard_payload, ensure_ascii=False, allow_nan=False)
    data_js_content = f"/* Real Audited NewSpace Dataset Payload (SHA-256 Verified) */\nwindow.NEWSPACE_DATA = {payload_json};\n"
    css_content = build_dashboard_css(PALETTE)
    plotly_bundle_src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    asset_ver = hashlib.sha256(
        (
            css_content
            + DATA_LOADER_JS
            + TOPSIS_ENGINE_JS
            + TABLE_RENDERER_JS
            + CHART_RENDERER_JS
            + UI_CONTROLLER_JS
        ).encode("utf-8")
    ).hexdigest()[:8]

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="{PALETTE.cosmic_bg}">
<title>NewSpace Analytics Cockpit — Benchmarking, Tendências &amp; Decisão Multicritério</title>
<link rel="icon" type="image/svg+xml" href="favicon.svg">
<link rel="alternate icon" href="{FAVICON_DATA_URI}">
<link rel="stylesheet" href="assets/css/dashboard.css?v={asset_ver}">
<script defer src="assets/vendor/plotly.min.js?v=4.1.1"></script>
<script defer src="assets/data/dashboard-data.js?v={asset_ver}"></script>
<script defer src="assets/js/data-loader.js?v={asset_ver}"></script>
<script defer src="assets/js/topsis-engine.js?v={asset_ver}"></script>
<script defer src="assets/js/table-renderer.js?v={asset_ver}"></script>
<script defer src="assets/js/chart-renderer.js?v={asset_ver}"></script>
<script defer src="assets/js/ui-controller.js?v={asset_ver}"></script>
</head>
<body>
<div class="space-backdrop" aria-hidden="true"></div>

<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="./" aria-label="NewSpace Analytics — início">
      <span class="brand-logo">{FAVICON_SVG}</span>
      <span class="brand-text">
        <span class="brand-title">NewSpace Analytics</span>
        <span class="brand-sub">Mineração, tendências &amp; risco atuarial</span>
      </span>
    </a>
    <a class="report-btn" href="relatorio.html">Relatório técnico <span aria-hidden="true">&rarr;</span></a>
  </div>
</header>

<section class="hero">
  <span class="eyebrow">Dados 100% reais e auditados</span>
  <h1 class="hero-title">Cockpit de decisão para operações NewSpace</h1>
  <p class="hero-sub">NASA JPL SBDB &amp; NHATS &middot; USSPACECOM NORAD SATCAT &middot; Harvard-Smithsonian GCAT</p>
</section>

<nav class="tabs" aria-label="Seções do painel">
  <div class="tabs-inner">
    <button type="button" class="tab-btn active" data-tab="tab-sim" title="Simulador Customizável TOPSIS (Prospecção)">Simulador TOPSIS</button>
    <button type="button" class="tab-btn" data-tab="tab-trends" title="Tendência de Produção de Informações (1957–2030)">Tendências 1957–2030</button>
    <button type="button" class="tab-btn" data-tab="tab-bench" title="Benchmarking da Mineração &amp; Motores">Benchmarking</button>
    <button type="button" class="tab-btn" data-tab="tab-actuarial" title="Corredores Orbitais &amp; Matriz Atuarial">Corredores &amp; Atuarial</button>
  </div>
</nav>

<main>
  <div id="app-error-banner" class="error-banner" role="alert"></div>

  <!-- TAB 1: CUSTOMIZABLE INTERACTIVE SIMULATOR -->
  <section id="tab-sim" class="tab-panel active">
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Registros reais auditados</div><div class="val">{total_records:,}</div><div class="hint">Zero dados fictícios (SHA-256)</div></div>
      <div class="kpi kpi-accent"><div class="lbl">Candidatos qualificados</div><div class="val" id="kpi-cand-count">-</div><div class="hint" id="kpi-retention">-</div></div>
      <div class="kpi"><div class="lbl">Alvo #1 recomendado (TOPSIS)</div><div class="val val-compact" id="kpi-top1-name">-</div><div class="hint" id="kpi-top1-meta">-</div></div>
      <div class="kpi"><div class="lbl">CAGR produção de informação</div><div class="val">{fc_2030['cagr_newspace_2015_2025_pct']:.1f}% <small>a.a.</small></div><div class="hint">Era NewSpace (2015–2025)</div></div>
    </div>

    <details class="criteria-intro-card">
      <summary>
        <h2 class="criteria-intro-title">O que são os Critérios de Seleção de Dados e sua Importância na Pesquisa Espacial</h2>
        <span class="summary-toggle" aria-hidden="true"></span>
      </summary>
      <p class="criteria-intro-lead">
        Na ciência de dados aeroespacial, os <strong>critérios de seleção de dados</strong> são filtros físicos, astrométricos e estatísticos aplicados sobre grandes catálogos brutos ({len(datasets.asteroids):,} asteroides próximos à Terra na base <em>NASA JPL SBDB</em> e {len(client_records):,} alvos com soluções balísticas completas no <em>NASA NHATS</em>) para isolar apenas objetos operacionalmente alcançáveis e com efemérides confiáveis. Como a produção de informações espaciais cresce exponencialmente mas apresenta forte assimetria observacional, essa triagem prévia impede que algoritmos de decisão (<em>MCDA-TOPSIS</em>) ranqueiem alvos energeticamente inviáveis ou com alto risco de perda navegacional.
      </p>
      <div class="criteria-pillars">
        <div class="criteria-pillar">
          <strong>1. Viabilidade Astrodinâmica (&Delta;v &amp; Janela)</strong>
          Restringe a análise a missões compatíveis com a capacidade propulsiva atual (&Delta;v &le; 6,50 km/s, comparável à transferência cislunar, e duração &le; 450 dias), reduzindo massa de propelente e custo de lançamento.
        </div>
        <div class="criteria-pillar">
          <strong>2. Precisão Navegacional (OCC &amp; Astrometria)</strong>
          Filtra incertezas orbitais elevadas ao exigir <em>Orbit Condition Code</em> OCC &le; 3 (escala NASA de 0 a 9) e &ge; 50 observações confirmadas, convertendo densidade de dados em segurança de guiagem e encontro orbital.
        </div>
        <div class="criteria-pillar">
          <strong>3. Aplicação Mineralógica, ISRU &amp; Atuarial</strong>
          Qualifica corpos com escala útil (&ge; 25 m) para sondas precursoras de reconhecimento espectral ou extração de recursos (água/voláteis no Complexo-C e metais em S/X), apoiando a subscrição securitária da missão.
        </div>
      </div>
    </details>

    <div class="sim-layout">
      <aside class="control-box">
        <button type="button" class="mobile-filter-toggle" id="btn-filter-toggle" aria-expanded="true">
          <span>Critérios &amp; pesos TOPSIS</span>
          <span id="filter-toggle-icon">&#9652; Ocultar</span>
        </button>
        <div class="ctrl-body" id="ctrl-body">
          <div class="ctrl-columns">
            <div class="ctrl-col">
              <h3>Critérios de Seleção de Dados</h3>
              <p class="ctrl-section-note">Restrições eliminatórias físicas, astrométricas e taxonômicas aplicadas sobre os alvos reais da NASA JPL.</p>
              <div class="ctrl-group">
                <label for="ctrl-dv"><span>&Delta;v máximo de missão (km/s)</span><strong id="lbl-dv">6.50</strong></label>
                <input type="range" id="ctrl-dv" min="4.0" max="11.0" step="0.25" value="6.5">
                <span class="ctrl-hint">Orçamento propulsivo mínimo ida e volta NHATS (padrão: &le; 6,50 km/s).</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-dur"><span>Duração máxima de voo (dias)</span><strong id="lbl-dur">450</strong></label>
                <input type="range" id="ctrl-dur" min="120" max="450" step="30" value="450">
                <span class="ctrl-hint">Janela operacional total da trajetória de menor &Delta;v (padrão: &le; 450 d).</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-occ"><span>Incerteza orbital máxima (OCC &le;)</span><strong id="lbl-occ">3</strong></label>
                <input type="range" id="ctrl-occ" min="0" max="9" step="1" value="3">
                <span class="ctrl-hint">Escala NASA de incerteza efeméride: 0 = órbita precisa a 9 = incerta (padrão: &le; 3).</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-obs"><span>Observações astrométricas mínimas</span><strong id="lbl-obs">50</strong></label>
                <input type="range" id="ctrl-obs" min="5" max="500" step="5" value="50">
                <span class="ctrl-hint">Volume mínimo de medições ópticas/radar no SBDB para rastreio seguro (padrão: &ge; 50).</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-diam"><span>Diâmetro físico mínimo (m)</span><strong id="lbl-diam">25</strong></label>
                <input type="range" id="ctrl-diam" min="5" max="400" step="5" value="25">
                <span class="ctrl-hint">Porte mínimo estimado/medido para operação robótica ou lavra (padrão: &ge; 25 m).</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-grp"><span>Filtro taxonômico / mineralógico</span></label>
                <select id="ctrl-grp">
                  <option value="ALL">Todos os Grupos (Incluindo Alvos p/ Sonda Precursora)</option>
                  <option value="KNOWN">Apenas Espectrografados Confirmados pela NASA</option>
                  <option value="Complexo-C">Complexo-C (Voláteis &amp; Água ISRU)</option>
                  <option value="Complexo-S">Complexo-S (Silicatados &amp; Rochosos)</option>
                  <option value="Complexo-X/M">Complexo-X/M (Metálicos / Grupo da Platina)</option>
                </select>
                <span class="ctrl-hint">Define o foco entre reconhecimento espectral precursor ou extração mineral dedicada.</span>
              </div>
            </div>

            <div class="ctrl-col">
              <h3>Pesos Decisórios MCDA-TOPSIS</h3>
              <p class="ctrl-section-note">Importância relativa na ordenação dos alvos qualificados (normalização automática para soma = 100%).</p>
              <div class="ctrl-group">
                <label for="ctrl-w1"><span>Eficiência astrodinâmica (&Delta;v)</span><strong id="lbl-w1">0.40</strong></label>
                <input type="range" id="ctrl-w1" min="0.05" max="1.0" step="0.05" value="0.40">
                <span class="ctrl-hint">Prioriza menor energia propulsiva combinada à menor duração de missão.</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-w2"><span>Maturidade da informação</span><strong id="lbl-w2">0.25</strong></label>
                <input type="range" id="ctrl-w2" min="0.05" max="1.0" step="0.05" value="0.25">
                <span class="ctrl-hint">Valoriza arco observacional longo, baixo OCC, espectroscopia e diâmetro medido.</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-w3"><span>Porte físico mineral</span><strong id="lbl-w3">0.20</strong></label>
                <input type="range" id="ctrl-w3" min="0.05" max="1.0" step="0.05" value="0.20">
                <span class="ctrl-hint">Valoriza maior volume potencial de recursos minerais/voláteis (escala logarítmica).</span>
              </div>
              <div class="ctrl-group">
                <label for="ctrl-w4"><span>Redundância de janelas</span><strong id="lbl-w4">0.15</strong></label>
                <input type="range" id="ctrl-w4" min="0.05" max="1.0" step="0.05" value="0.15">
                <span class="ctrl-hint">Valoriza quantidade de soluções balísticas alternativas para flexibilidade de lançamento.</span>
              </div>

              <div class="ctrl-actions">
                <button type="button" class="action-btn" id="btn-export-csv">Exportar seleção (CSV)</button>
                <button type="button" class="action-btn secondary" id="btn-reset-defaults">Restaurar critérios padrão</button>
              </div>
            </div>
          </div>
        </div>
      </aside>

      <div class="sim-results">
        <div class="chart-card">
          <div class="section-subtitle" id="sim-scatter-title">Espaço de Decisão Astrodinâmico — Alvos Qualificados (NASA JPL NHATS)</div>
          <div id="plot-sim-scatter" class="plot-box"></div>
        </div>
        <div class="chart-card">
          <div class="section-subtitle">Top 20 Asteroides Próximos à Terra Ranqueados em Tempo Real (NASA JPL NHATS)</div>
          <div class="mobile-scroll-hint">&harr; Deslize lateralmente para ver todas as métricas</div>
          <div class="table-wrap table-scroll-y">
            <table class="ranking-tbl">
              <thead>
                <tr>
                  <th class="txt-col">Rank</th>
                  <th class="txt-col">Designação</th>
                  <th class="txt-col">Asteroide Oficial (NASA JPL)</th>
                  <th class="txt-col">Espectro</th>
                  <th class="num-col">&Delta;v Mín. (km/s)</th>
                  <th class="num-col">Duração (d)</th>
                  <th class="num-col">Diâm. Ref. (m)</th>
                  <th class="num-col">Obs.</th>
                  <th class="num-col">OCC</th>
                  <th class="num-col">Score TOPSIS</th>
                </tr>
              </thead>
              <tbody id="tbl-sim-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 2: INFORMATION PRODUCTION TRENDS -->
  <section id="tab-trends" class="tab-panel">
    <div class="chart-card">
      <div class="section-subtitle">Tendência Empírica de Produção de Informações Espaciais e Projeção 2030</div>
      <div id="plot-trends" class="plot-box"></div>
    </div>
    <div class="grid-2">
      <div class="chart-card">
        <div class="section-subtitle">Lei Informação &times; Risco: Observações por Nível de Incerteza OCC</div>
        <div id="plot-obs-law" class="plot-box"></div>
      </div>
      <div class="chart-card">
        <div class="section-subtitle">Bifurcação de Tráfego nos Corredores Orbitais (Ativos vs Detritos)</div>
        <div id="plot-corridors-tr" class="plot-box"></div>
      </div>
    </div>
    <div class="chart-card">
      <div class="section-subtitle">Comparativo Estrutural de Produção de Informações entre as Eras Espaciais (1957–2025)</div>
      {era_html_table}
    </div>
  </section>

  <!-- TAB 3: DATA MINING BENCHMARK -->
  <section id="tab-bench" class="tab-panel">
    <div class="kpi-row kpi-3">
      <div class="kpi kpi-accent"><div class="lbl">Motor analítico mais rápido</div><div class="val val-compact">{best_eng['engine']}</div><div class="hint">Latência total: {best_eng['latencia_total_mediana_ms']:.2f} ms ({best_eng['speedup_vs_sqlite_sem_indice']:.1f}x speedup)</div></div>
      <div class="kpi"><div class="lbl">Melhor clusterização NHATS</div><div class="val val-compact">{bench.clustering_benchmark.iloc[0]['algoritmo']}</div><div class="hint">Silhouette: {bench.clustering_benchmark.iloc[0]['silhouette_score']:.4f}</div></div>
      <div class="kpi"><div class="lbl">Melhor classificador atuarial</div><div class="val val-compact">{SHORT_MODEL_LABELS.get(bench.classification_benchmark.iloc[0]['modelo_atuarial'], bench.classification_benchmark.iloc[0]['modelo_atuarial'])}</div><div class="hint">ROC-AUC: {bench.classification_benchmark.iloc[0]['roc_auc']:.4f} | Brier: {bench.classification_benchmark.iloc[0]['brier_score_calibracao']:.5f}</div></div>
    </div>
    <div class="grid-2">
      <div class="chart-card">
        <div class="section-subtitle">Desempenho de Consulta: Motores Analíticos &amp; Relacionais</div>
        <div id="plot-eng-lat" class="plot-box"></div>
      </div>
      <div class="chart-card">
        <div class="section-subtitle">Eficiência de Armazenamento &amp; Compressão Física</div>
        <div id="plot-eng-sto" class="plot-box"></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="chart-card">
        <div class="section-subtitle">Modelos Preditivos de Sinistro Atuarial (7.418 Voos — 5-Fold CV)</div>
        <div id="plot-ml-clf" class="plot-box"></div>
      </div>
      <div class="chart-card">
        <div class="section-subtitle">Importância Empírica dos Fatores de Risco de Lançamento</div>
        <div id="plot-ml-imp" class="plot-box"></div>
      </div>
    </div>
  </section>

  <!-- TAB 4: ORBITAL CORRIDORS & ACTUARIAL MATRIX -->
  <section id="tab-actuarial" class="tab-panel">
    <div class="grid-2">
      <div class="chart-card">
        <div class="section-subtitle">Matriz Integrada de Subscrição Atuarial: Retorno TOPSIS vs Risco Operacional</div>
        <div id="plot-actuarial-mat" class="plot-box"></div>
      </div>
      <div class="chart-card">
        <div class="section-subtitle">Bifurcação de Tráfego nos Corredores Orbitais (NORAD SATCAT)</div>
        <div id="plot-corridors-act" class="plot-box"></div>
      </div>
    </div>
    <div class="chart-card">
      <div class="section-subtitle">Cenários Integrados de Subscrição Atuarial (Alvo &times; Veículo &times; Corredor Cislunar)</div>
      {scen_html_table}
    </div>
  </section>
</main>

<footer class="site-footer">
  <span>NewSpace Analytics &middot; {total_records:,} registros reais auditados</span>
  <a href="relatorio.html">Relatório técnico executivo</a>
</footer>
</body>
</html>
"""

    out_path = _write_modular_dashboard_bundle(
        reports_dir,
        "newspace_dashboard.html",
        html,
        css_content,
        data_js_content,
        plotly_bundle_src,
    )

    if DOCS_DIR.exists():
        _write_modular_dashboard_bundle(
            DOCS_DIR,
            "index.html",
            html,
            css_content,
            data_js_content,
            plotly_bundle_src,
        )

    return out_path
