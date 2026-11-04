"""Customizable Visual Analytics, Notebook ipywidgets Cockpit & Standalone HTML5 Dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
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
    """Apply balanced margins, auto-margin axes, and official dark cosmic palette."""
    legend_cfg = (
        dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(35, 20, 91, 0.60)",
            bordercolor="rgba(28, 163, 158, 0.35)",
            borderwidth=1,
            font=dict(size=11.5),
        )
        if top_legend
        else dict(
            bgcolor="rgba(35, 20, 91, 0.60)",
            bordercolor="rgba(28, 163, 158, 0.35)",
            borderwidth=1,
            font=dict(size=11.5),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15.5, color=PALETTE.text_primary, family="Segoe UI, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor=PALETTE.cosmic_bg,
        plot_bgcolor="#1D1226",
        font=dict(color=PALETTE.text_primary, family="Segoe UI, sans-serif", size=12),
        height=height,
        margin=dict(l=24, r=28, t=76, b=44),
        legend=legend_cfg,
        hoverlabel=dict(
            bgcolor=PALETTE.space_violet,
            bordercolor=PALETTE.vibrant_cyan,
            font=dict(color="#FFFFFF", size=12),
        ),
    )
    fig.update_xaxes(gridcolor=PALETTE.grid_line, zerolinecolor=PALETTE.grid_line, automargin=True)
    fig.update_yaxes(gridcolor=PALETTE.grid_line, zerolinecolor=PALETTE.grid_line, automargin=True)
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
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.5)", width=0.8)),
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
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.5)", width=0.8)),
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
            marker=dict(color=PALETTE.vibrant_cyan, line=dict(color="rgba(255,255,255,0.5)", width=0.7)),
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
            marker=dict(color=PALETTE.deep_teal, line=dict(color="rgba(255,255,255,0.5)", width=0.7)),
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
            marker=dict(color=PALETTE.deep_teal, line=dict(color="rgba(255,255,255,0.45)", width=0.7)),
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
            fillcolor="rgba(28, 163, 158, 0.20)",
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
            fillcolor="rgba(28, 163, 158, 0.22)",
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
        bgcolor="rgba(35, 20, 91, 0.88)",
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
                line=dict(width=1.0, color="rgba(255,255,255,0.65)"),
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


def export_standalone_custom_dashboard(
    datasets: MinedDatasets,
    bench: BenchmarkSuiteResults,
    trends: TrendAnalysisSummary,
    selection: SelectionDecisionOutput,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Export interactive, overlap-free, proportionally styled HTML5 cockpit + sync to docs/."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "newspace_dashboard.html"
    (reports_dir / "favicon.svg").write_text(FAVICON_SVG, encoding="utf-8")

    # Dedicated individual proportional figures for responsive 2-column CSS grid
    f_eng_lat_json = fig_engine_latency_card(bench).to_json()
    f_eng_sto_json = fig_storage_footprint_card(bench).to_json()
    f_ml_clf_json = fig_ml_classifiers_card(bench).to_json()
    f_ml_imp_json = fig_risk_factors_card(bench).to_json()
    f_tr_json = fig_information_trends(trends).to_json()
    f_obs_json = fig_observability_card(trends).to_json()
    f_cor_json = fig_corridors_card(datasets).to_json()
    f_ac_json = fig_actuarial_scenarios_matrix(selection).to_json()

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

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NewSpace Analytics Cockpit — Benchmarking, Tendências & Decisão Multicritério</title>
<link rel="icon" type="image/svg+xml" href="favicon.svg">
<link rel="alternate icon" href="{FAVICON_DATA_URI}">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: {PALETTE.cosmic_bg};
    --violet: {PALETTE.space_violet};
    --blue: {PALETTE.orbital_blue};
    --teal: {PALETTE.deep_teal};
    --cyan: {PALETTE.vibrant_cyan};
    --text: {PALETTE.text_primary};
    --muted: {PALETTE.text_muted};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }}
  header {{
    background: linear-gradient(90deg, var(--bg) 0%, var(--violet) 48%, var(--blue) 100%);
    border-bottom: 2px solid var(--cyan);
    padding: 16px 28px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;
  }}
  .brand-wrap {{ display: flex; align-items: center; gap: 14px; }}
  .brand-logo {{ width: 42px; height: 42px; flex-shrink: 0; }}
  .brand-title {{ font-size: 20px; font-weight: 800; letter-spacing: 0.02em; }}
  .brand-sub {{ font-size: 12px; color: #cbd5e1; margin-top: 2px; }}
  .header-right {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  .palette-pills {{ display: flex; gap: 6px; align-items: center; font-size: 11px; color: #cbd5e1; }}
  .swatch {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.4); }}
  .report-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(28, 163, 158, 0.20); color: #fff; border: 1px solid var(--cyan);
    padding: 8px 15px; border-radius: 8px; text-decoration: none; font-size: 12.5px; font-weight: 700;
    transition: all 0.15s;
  }}
  .report-btn:hover {{ background: var(--cyan); color: #06181c; }}
  nav.tabs {{
    display: flex; gap: 8px; padding: 12px 28px; background: #1c1124; border-bottom: 1px solid #2d223b; flex-wrap: wrap;
  }}
  .tab-btn {{
    background: rgba(35, 20, 91, 0.55); color: var(--text); border: 1px solid rgba(28, 163, 158, 0.35);
    padding: 9px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px; transition: all 0.15s;
  }}
  .tab-btn:hover {{ background: var(--blue); }}
  .tab-btn.active {{ background: var(--cyan); color: #091318; border-color: #fff; font-weight: 800; }}
  main {{ padding: 22px 28px; max-width: 1560px; margin: 0 auto; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 20px; }}
  .kpi {{
    background: linear-gradient(135deg, rgba(35,20,91,0.75), rgba(9,69,108,0.65));
    border: 1px solid rgba(28,163,158,0.42); border-radius: 10px; padding: 14px 18px;
  }}
  .kpi .lbl {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi .val {{ font-size: 23px; font-weight: 800; color: var(--cyan); margin-top: 4px; }}
  .kpi .hint {{ font-size: 11.5px; color: #cbd5e1; margin-top: 3px; }}
  .grid-2 {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 20px; margin-bottom: 20px;
  }}
  @media (max-width: 640px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .chart-card {{
    background: #1D1226; border: 1px solid rgba(28,163,158,0.28); border-radius: 12px;
    padding: 14px 16px; margin-bottom: 20px; box-shadow: 0 10px 26px rgba(0,0,0,0.35);
    overflow: hidden;
  }}
  .grid-2 .chart-card {{ margin-bottom: 0; }}
  .section-subtitle {{
    margin: 4px 4px 12px; color: var(--cyan); font-size: 15.5px; font-weight: 700;
  }}
  .sim-layout {{ display: grid; grid-template-columns: 360px 1fr; gap: 20px; align-items: start; }}
  @media (max-width: 1024px) {{ .sim-layout {{ grid-template-columns: 1fr; }} }}
  .control-box {{
    background: linear-gradient(180deg, rgba(35,20,91,0.78), rgba(22,13,24,0.95));
    border: 1px solid var(--cyan); border-radius: 12px; padding: 18px;
  }}
  .control-box h3 {{ margin: 0 0 12px; font-size: 15px; color: var(--cyan); border-bottom: 1px solid rgba(28,163,158,0.3); padding-bottom: 6px; }}
  .ctrl-group {{ margin-bottom: 12px; }}
  .ctrl-group label {{ display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: #e2e8f0; }}
  .ctrl-group input[type=range], .ctrl-group select {{ width: 100%; accent-color: var(--cyan); }}
  .ctrl-group select {{
    background: var(--bg); color: #fff; border: 1px solid var(--teal); padding: 6px 8px; border-radius: 6px;
  }}
  .action-btn {{
    width: 100%; padding: 10px; background: var(--cyan); color: #07191e; border: none;
    border-radius: 8px; font-weight: 800; cursor: pointer; margin-top: 8px; font-size: 13px;
  }}
  .action-btn.secondary {{ background: var(--blue); color: #fff; border: 1px solid var(--cyan); }}
  .table-wrap {{
    width: 100%; overflow-x: auto; border-radius: 8px; border: 1px solid rgba(28, 163, 158, 0.22);
  }}
  table.ranking-tbl, table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12.8px; }}
  table.ranking-tbl th, table.data-table th {{
    background: var(--blue); color: #fff; padding: 10px 12px; font-weight: 700; position: sticky; top: 0;
    border-bottom: 2px solid var(--cyan);
  }}
  table.ranking-tbl td, table.data-table td {{
    padding: 8.5px 12px; border-bottom: 1px solid rgba(148,163,184,0.16);
  }}
  .txt-col {{ text-align: left; }}
  .num-col {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  table.ranking-tbl tr:nth-child(even), table.data-table tr:nth-child(even) {{ background: rgba(35,20,91,0.22); }}
  table.ranking-tbl tr:hover, table.data-table tr:hover {{ background: rgba(28,163,158,0.14); }}
</style>
</head>
<body>
<header>
  <div class="brand-wrap">
    <div class="brand-logo">{FAVICON_SVG}</div>
    <div>
      <div class="brand-title">NEWSPACE ANALYTICS COCKPIT • MINERAÇÃO &amp; RISCO ATUARIAL</div>
      <div class="brand-sub">Dados 100% Reais Auditados (NASA JPL SBDB &amp; NHATS • USSPACECOM NORAD SATCAT • Harvard-Smithsonian GCAT)</div>
    </div>
  </div>
  <div class="header-right">
    <div class="palette-pills">
      <span>Paleta:</span>
      <span class="swatch" style="background:#160D18" title="#160D18"></span>
      <span class="swatch" style="background:#23145B" title="#23145B"></span>
      <span class="swatch" style="background:#09456C" title="#09456C"></span>
      <span class="swatch" style="background:#026F6E" title="#026F6E"></span>
      <span class="swatch" style="background:#1CA39E" title="#1CA39E"></span>
    </div>
    <a class="report-btn" href="relatorio.html">Relatório Técnico Executivo &rarr;</a>
  </div>
</header>

<nav class="tabs">
  <button class="tab-btn active" onclick="switchTab('tab-sim', this)">1. Simulador Customizável TOPSIS (Prospecção)</button>
  <button class="tab-btn" onclick="switchTab('tab-trends', this)">2. Tendência de Produção de Informações (1957–2030)</button>
  <button class="tab-btn" onclick="switchTab('tab-bench', this)">3. Benchmarking da Mineração &amp; Motores</button>
  <button class="tab-btn" onclick="switchTab('tab-actuarial', this)">4. Corredores Orbitais &amp; Matriz Atuarial</button>
</nav>

<main>
  <!-- TAB 1: CUSTOMIZABLE INTERACTIVE SIMULATOR -->
  <section id="tab-sim" class="tab-panel active">
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Registros Reais Auditados</div><div class="val">{total_records:,}</div><div class="hint">Zero dados fictícios (SHA-256)</div></div>
      <div class="kpi"><div class="lbl">Candidatos Qualificados (Filtro Atual)</div><div class="val" id="kpi-cand-count">-</div><div class="hint" id="kpi-retention">-</div></div>
      <div class="kpi"><div class="lbl">Alvo #1 Recomendado (TOPSIS)</div><div class="val" id="kpi-top1-name" style="font-size:18px;">-</div><div class="hint" id="kpi-top1-meta">-</div></div>
      <div class="kpi"><div class="lbl">CAGR Produção de Informação</div><div class="val">{fc_2030['cagr_newspace_2015_2025_pct']:.1f}% a.a.</div><div class="hint">Era NewSpace (2015–2025)</div></div>
    </div>

    <div class="sim-layout">
      <aside class="control-box">
        <h3>Critérios de Seleção de Dados</h3>
        <div class="ctrl-group">
          <label><span>Δv Máximo de Missão (km/s)</span><strong id="lbl-dv">6.50</strong></label>
          <input type="range" id="ctrl-dv" min="4.0" max="11.0" step="0.25" value="6.5" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Duração Máxima (dias)</span><strong id="lbl-dur">450</strong></label>
          <input type="range" id="ctrl-dur" min="120" max="450" step="30" value="450" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Incerteza Orbital Máxima (OCC ≤)</span><strong id="lbl-occ">3</strong></label>
          <input type="range" id="ctrl-occ" min="0" max="9" step="1" value="3" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Observações Astrométricas Mín.</span><strong id="lbl-obs">50</strong></label>
          <input type="range" id="ctrl-obs" min="5" max="500" step="5" value="50" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Diâmetro Físico Mínimo (m)</span><strong id="lbl-diam">25</strong></label>
          <input type="range" id="ctrl-diam" min="5" max="400" step="5" value="25" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Filtro Taxonômico / Mineralógico</span></label>
          <select id="ctrl-grp" onchange="recomputeTOPSIS()">
            <option value="ALL">Todos os Grupos (Incluindo Alvos p/ Sonda Precursora)</option>
            <option value="KNOWN">Apenas Espectrografados Confirmados pela NASA</option>
            <option value="Complexo-C">Complexo-C (Voláteis &amp; Água ISRU)</option>
            <option value="Complexo-S">Complexo-S (Silicatados &amp; Rochosos)</option>
            <option value="Complexo-X/M">Complexo-X/M (Metálicos / Grupo da Platina)</option>
          </select>
        </div>

        <h3 style="margin-top:18px;">Pesos Decisórios MCDA-TOPSIS</h3>
        <div class="ctrl-group">
          <label><span>Peso Eficiência Astrodinâmica (Δv)</span><strong id="lbl-w1">0.40</strong></label>
          <input type="range" id="ctrl-w1" min="0.05" max="1.0" step="0.05" value="0.40" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Peso Maturidade da Informação</span><strong id="lbl-w2">0.25</strong></label>
          <input type="range" id="ctrl-w2" min="0.05" max="1.0" step="0.05" value="0.25" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Peso Porte Físico Mineral</span><strong id="lbl-w3">0.20</strong></label>
          <input type="range" id="ctrl-w3" min="0.05" max="1.0" step="0.05" value="0.20" oninput="recomputeTOPSIS()">
        </div>
        <div class="ctrl-group">
          <label><span>Peso Redundância de Janelas</span><strong id="lbl-w4">0.15</strong></label>
          <input type="range" id="ctrl-w4" min="0.05" max="1.0" step="0.05" value="0.15" oninput="recomputeTOPSIS()">
        </div>

        <button class="action-btn" onclick="exportFilteredCSV()">Exportar Seleção Qualificada (CSV)</button>
        <button class="action-btn secondary" onclick="resetDefaults()">Restaurar Critérios Padrão da Equipe</button>
      </aside>

      <div>
        <div class="chart-card"><div id="plot-sim-scatter" style="height:470px;"></div></div>
        <div class="chart-card">
          <div class="section-subtitle">Top 20 Asteroides Próximos à Terra Ranqueados em Tempo Real (NASA JPL NHATS)</div>
          <div class="table-wrap" style="max-height:410px; overflow-y:auto;">
            <table class="ranking-tbl">
              <thead>
                <tr>
                  <th class="txt-col">Rank</th>
                  <th class="txt-col">Designação</th>
                  <th class="txt-col">Asteroide Oficial (NASA JPL)</th>
                  <th class="txt-col">Espectro</th>
                  <th class="num-col">Δv Mín. (km/s)</th>
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
    <div class="chart-card"><div id="plot-trends"></div></div>
    <div class="grid-2">
      <div class="chart-card"><div id="plot-obs-law"></div></div>
      <div class="chart-card"><div id="plot-corridors-tr"></div></div>
    </div>
    <div class="chart-card">
      <div class="section-subtitle">Comparativo Estrutural de Produção de Informações entre as Eras Espaciais (1957–2025)</div>
      {era_html_table}
    </div>
  </section>

  <!-- TAB 3: DATA MINING BENCHMARK -->
  <section id="tab-bench" class="tab-panel">
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Motor Analítico Mais Rápido</div><div class="val" style="font-size:19px;">{best_eng['engine']}</div><div class="hint">Latência total: {best_eng['latencia_total_mediana_ms']:.2f} ms ({best_eng['speedup_vs_sqlite_sem_indice']:.1f}x speedup)</div></div>
      <div class="kpi"><div class="lbl">Melhor Clusterização NHATS</div><div class="val" style="font-size:19px;">{bench.clustering_benchmark.iloc[0]['algoritmo']}</div><div class="hint">Silhouette: {bench.clustering_benchmark.iloc[0]['silhouette_score']:.4f}</div></div>
      <div class="kpi"><div class="lbl">Melhor Classificador Atuarial</div><div class="val" style="font-size:19px;">{SHORT_MODEL_LABELS.get(bench.classification_benchmark.iloc[0]['modelo_atuarial'], bench.classification_benchmark.iloc[0]['modelo_atuarial'])}</div><div class="hint">ROC-AUC: {bench.classification_benchmark.iloc[0]['roc_auc']:.4f} | Brier: {bench.classification_benchmark.iloc[0]['brier_score_calibracao']:.5f}</div></div>
    </div>
    <div class="grid-2">
      <div class="chart-card"><div id="plot-eng-lat"></div></div>
      <div class="chart-card"><div id="plot-eng-sto"></div></div>
    </div>
    <div class="grid-2">
      <div class="chart-card"><div id="plot-ml-clf"></div></div>
      <div class="chart-card"><div id="plot-ml-imp"></div></div>
    </div>
  </section>

  <!-- TAB 4: ORBITAL CORRIDORS & ACTUARIAL MATRIX -->
  <section id="tab-actuarial" class="tab-panel">
    <div class="grid-2">
      <div class="chart-card"><div id="plot-actuarial-mat"></div></div>
      <div class="chart-card"><div id="plot-corridors-act"></div></div>
    </div>
    <div class="chart-card">
      <div class="section-subtitle">Cenários Integrados de Subscrição Atuarial (Alvo × Veículo × Corredor Cislunar)</div>
      {scen_html_table}
    </div>
  </section>
</main>

<script>
const NHATS_DATA = {json.dumps(client_records, ensure_ascii=False)};
const SPEC_ENG_LAT = {f_eng_lat_json};
const SPEC_ENG_STO = {f_eng_sto_json};
const SPEC_ML_CLF = {f_ml_clf_json};
const SPEC_ML_IMP = {f_ml_imp_json};
const SPEC_TR = {f_tr_json};
const SPEC_OBS = {f_obs_json};
const SPEC_COR = {f_cor_json};
const SPEC_AC = {f_ac_json};

let currentRanked = [];

function renderStaticPlots() {{
  const cfg = {{responsive: true, displaylogo: false}};
  Plotly.newPlot('plot-eng-lat', SPEC_ENG_LAT.data, SPEC_ENG_LAT.layout, cfg);
  Plotly.newPlot('plot-eng-sto', SPEC_ENG_STO.data, SPEC_ENG_STO.layout, cfg);
  Plotly.newPlot('plot-ml-clf', SPEC_ML_CLF.data, SPEC_ML_CLF.layout, cfg);
  Plotly.newPlot('plot-ml-imp', SPEC_ML_IMP.data, SPEC_ML_IMP.layout, cfg);
  Plotly.newPlot('plot-trends', SPEC_TR.data, SPEC_TR.layout, cfg);
  Plotly.newPlot('plot-obs-law', SPEC_OBS.data, SPEC_OBS.layout, cfg);
  Plotly.newPlot('plot-corridors-tr', SPEC_COR.data, SPEC_COR.layout, cfg);
  Plotly.newPlot('plot-corridors-act', SPEC_COR.data, SPEC_COR.layout, cfg);
  Plotly.newPlot('plot-actuarial-mat', SPEC_AC.data, SPEC_AC.layout, cfg);
}}

function switchTab(tabId, btn) {{
  document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}

function recomputeTOPSIS() {{
  const maxDv = parseFloat(document.getElementById('ctrl-dv').value);
  const maxDur = parseInt(document.getElementById('ctrl-dur').value);
  const maxOcc = parseInt(document.getElementById('ctrl-occ').value);
  const minObs = parseInt(document.getElementById('ctrl-obs').value);
  const minDiam = parseFloat(document.getElementById('ctrl-diam').value);
  const grpFilter = document.getElementById('ctrl-grp').value;

  const w1 = parseFloat(document.getElementById('ctrl-w1').value);
  const w2 = parseFloat(document.getElementById('ctrl-w2').value);
  const w3 = parseFloat(document.getElementById('ctrl-w3').value);
  const w4 = parseFloat(document.getElementById('ctrl-w4').value);

  document.getElementById('lbl-dv').textContent = maxDv.toFixed(2);
  document.getElementById('lbl-dur').textContent = maxDur;
  document.getElementById('lbl-occ').textContent = maxOcc;
  document.getElementById('lbl-obs').textContent = minObs;
  document.getElementById('lbl-diam').textContent = minDiam;
  document.getElementById('lbl-w1').textContent = w1.toFixed(2);
  document.getElementById('lbl-w2').textContent = w2.toFixed(2);
  document.getElementById('lbl-w3').textContent = w3.toFixed(2);
  document.getElementById('lbl-w4').textContent = w4.toFixed(2);

  let cand = NHATS_DATA.filter(d =>
    d.dv <= maxDv &&
    d.dur <= maxDur &&
    d.occ <= maxOcc &&
    d.obs >= minObs &&
    d.diam >= minDiam &&
    (grpFilter === 'ALL' ||
     (grpFilter === 'KNOWN' && d.has_spec === 1) ||
     d.grp.startsWith(grpFilter))
  );

  document.getElementById('kpi-cand-count').textContent = cand.length.toLocaleString('pt-BR');
  document.getElementById('kpi-retention').textContent =
    ((100.0 * cand.length / NHATS_DATA.length).toFixed(2)) + '% dos 7.169 alvos NHATS';

  const tbody = document.getElementById('tbl-sim-body');
  if (cand.length === 0) {{
    document.getElementById('kpi-top1-name').textContent = 'Nenhum candidato';
    document.getElementById('kpi-top1-meta').textContent = 'Relaxe os filtros ao lado';
    tbody.innerHTML = '<tr><td colspan="10" class="txt-col">Nenhum asteroide atende simultaneamente às restrições atuais.</td></tr>';
    Plotly.purge('plot-sim-scatter');
    currentRanked = [];
    return;
  }}

  const maxLogObs = Math.max(...cand.map(d => Math.log1p(d.obs)), 1.0);
  const maxLogArc = Math.max(...cand.map(d => Math.log1p(d.arc)), 1.0);

  const feats = cand.map(d => {{
    const infoIdx = 100.0 * (
      0.35 * (Math.log1p(d.obs) / maxLogObs) +
      0.25 * (Math.log1p(d.arc) / maxLogArc) +
      0.25 * ((9.0 - Math.min(9, Math.max(0, d.occ))) / 9.0) +
      0.10 * d.has_spec +
      0.05 * d.diam_m_real
    );
    const c1 = 1.0 / (d.dv * Math.sqrt(Math.max(d.dur, 1.0)));
    const c2 = infoIdx;
    const c3 = Math.log1p(d.diam);
    const c4 = Math.log1p(d.traj);
    return {{...d, infoIdx, c1, c2, c3, c4}};
  }});

  const norm = k => Math.sqrt(feats.reduce((acc, r) => acc + r[k]*r[k], 0)) || 1.0;
  const n1 = norm('c1'), n2 = norm('c2'), n3 = norm('c3'), n4 = norm('c4');
  const wSum = (w1 + w2 + w3 + w4) || 1.0;
  const nw1 = w1/wSum, nw2 = w2/wSum, nw3 = w3/wSum, nw4 = w4/wSum;

  feats.forEach(r => {{
    r.v1 = (r.c1 / n1) * nw1;
    r.v2 = (r.c2 / n2) * nw2;
    r.v3 = (r.c3 / n3) * nw3;
    r.v4 = (r.c4 / n4) * nw4;
  }});

  const ib1 = Math.max(...feats.map(r => r.v1)), iw1 = Math.min(...feats.map(r => r.v1));
  const ib2 = Math.max(...feats.map(r => r.v2)), iw2 = Math.min(...feats.map(r => r.v2));
  const ib3 = Math.max(...feats.map(r => r.v3)), iw3 = Math.min(...feats.map(r => r.v3));
  const ib4 = Math.max(...feats.map(r => r.v4)), iw4 = Math.min(...feats.map(r => r.v4));

  feats.forEach(r => {{
    const db = Math.hypot(r.v1 - ib1, r.v2 - ib2, r.v3 - ib3, r.v4 - ib4);
    const dw = Math.hypot(r.v1 - iw1, r.v2 - iw2, r.v3 - iw3, r.v4 - iw4);
    r.topsis = (db + dw) > 0 ? (100.0 * dw / (db + dw)) : 50.0;
  }});

  feats.sort((a, b) => (b.topsis - a.topsis) || (a.dv - b.dv));
  currentRanked = feats;

  const top1 = feats[0];
  document.getElementById('kpi-top1-name').textContent = top1.name;
  document.getElementById('kpi-top1-meta').textContent =
    `Δv ${{top1.dv.toFixed(3)}} km/s • TOPSIS ${{top1.topsis.toFixed(1)}} • Espectro: ${{top1.spec}}`;

  const topPlot = feats.slice(0, 300);
  const trace = {{
    x: topPlot.map(r => r.dv),
    y: topPlot.map(r => r.dur),
    mode: 'markers',
    type: 'scatter',
    text: topPlot.map((r, idx) =>
      `<b>#${{idx+1}} ${{r.name}}</b><br>Espectro: ${{r.spec}}<br>Diâmetro Ref.: ${{r.diam}} m<br>Observações: ${{r.obs}} (OCC ${{r.occ}})<br>TOPSIS: ${{r.topsis.toFixed(2)}}`
    ),
    hovertemplate: '%{{text}}<br>Δv: %{{x:.3f}} km/s | Duração: %{{y}} d<extra></extra>',
    marker: {{
      size: topPlot.map(r => Math.min(30, Math.max(8, Math.log1p(r.diam) * 4.0))),
      color: topPlot.map(r => r.topsis),
      colorscale: [[0,'#23145B'],[0.35,'#09456C'],[0.7,'#026F6E'],[1,'#1CA39E']],
      colorbar: {{title: 'Score<br>TOPSIS'}},
      line: {{color: 'rgba(255,255,255,0.6)', width: 1}}
    }}
  }};

  const layout = {{
    title: {{text: `<b>Espaço de Decisão Astrodinâmico — Top ${{topPlot.length}} Alvos Qualificados</b>`, font: {{color: '#F1F5F9', size: 15.5}}}},
    paper_bgcolor: '#1D1226', plot_bgcolor: '#160D18',
    font: {{color: '#F1F5F9', family: 'Segoe UI, sans-serif'}},
    margin: {{l: 55, r: 30, t: 54, b: 48}},
    xaxis: {{title: 'Δv Mínimo NASA NHATS (km/s)', gridcolor: '#2A2438', automargin: true}},
    yaxis: {{title: 'Duração da Missão (Dias)', gridcolor: '#2A2438', automargin: true}}
  }};
  Plotly.react('plot-sim-scatter', [trace], layout, {{responsive: true, displaylogo: false}});

  tbody.innerHTML = feats.slice(0, 20).map((r, idx) => `
    <tr>
      <td class="txt-col"><strong>#${{idx + 1}}</strong></td>
      <td class="txt-col">${{r.pdes}}</td>
      <td class="txt-col">${{r.name}}</td>
      <td class="txt-col">${{r.spec}}</td>
      <td class="num-col" style="color:#34D399;font-weight:700;">${{r.dv.toFixed(3)}}</td>
      <td class="num-col">${{r.dur}}</td>
      <td class="num-col">${{r.diam.toFixed(1)}}</td>
      <td class="num-col">${{r.obs.toLocaleString('pt-BR')}}</td>
      <td class="num-col">${{r.occ}}</td>
      <td class="num-col" style="color:#1CA39E;font-weight:800;">${{r.topsis.toFixed(2)}}</td>
    </tr>
  `).join('');
}}

function resetDefaults() {{
  document.getElementById('ctrl-dv').value = 6.5;
  document.getElementById('ctrl-dur').value = 450;
  document.getElementById('ctrl-occ').value = 3;
  document.getElementById('ctrl-obs').value = 50;
  document.getElementById('ctrl-diam').value = 25;
  document.getElementById('ctrl-grp').value = 'ALL';
  document.getElementById('ctrl-w1').value = 0.40;
  document.getElementById('ctrl-w2').value = 0.25;
  document.getElementById('ctrl-w3').value = 0.20;
  document.getElementById('ctrl-w4').value = 0.15;
  recomputeTOPSIS();
}}

function exportFilteredCSV() {{
  if (!currentRanked.length) return;
  const header = ['rank','spkid','designacao','asteroide_oficial','espectro','delta_v_kms','duracao_dias','diametro_ref_m','observacoes','occ','score_topsis'];
  const rows = currentRanked.map((r, i) => [
    i+1, r.spkid, `"${{r.pdes}}"`, `"${{r.name}}"`, `"${{r.spec}}"`,
    r.dv, r.dur, r.diam, r.obs, r.occ, r.topsis.toFixed(2)
  ].join(','));
  const blob = new Blob([[header.join(','), ...rows].join('\\n')], {{type: 'text/csv;charset=utf-8;'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'newspace_selecao_topsis_qualificada.csv';
  a.click();
}}

window.addEventListener('DOMContentLoaded', () => {{
  renderStaticPlots();
  recomputeTOPSIS();
}});
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")

    # Sync directly to docs/index.html and docs/favicon.svg when docs/ exists
    if DOCS_DIR.exists():
        (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
        (DOCS_DIR / "favicon.svg").write_text(FAVICON_SVG, encoding="utf-8")

    return out_path
