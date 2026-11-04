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
from newspace.config import DEFAULT_CRITERIA, PALETTE, REPORTS_DIR
from newspace.ingestion import MinedDatasets
from newspace.selection_trends import (
    SelectionDecisionOutput,
    TrendAnalysisSummary,
    apply_selection_criteria_and_topsis,
)


def _apply_cosmic_layout(fig: go.Figure, title: str, height: int = 500) -> go.Figure:
    """Apply the team's official NewSpace dark cosmic palette (#160D18 .. #1CA39E) consistently."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=17, color=PALETTE.text_primary, family="Segoe UI, sans-serif"),
            x=0.02,
        ),
        paper_bgcolor=PALETTE.cosmic_bg,
        plot_bgcolor="#1D1226",
        font=dict(color=PALETTE.text_primary, family="Segoe UI, sans-serif", size=12),
        height=height,
        margin=dict(l=55, r=35, t=68, b=50),
        legend=dict(
            bgcolor="rgba(35, 20, 91, 0.55)",
            bordercolor="rgba(28, 163, 158, 0.35)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor=PALETTE.space_violet,
            bordercolor=PALETTE.vibrant_cyan,
            font=dict(color="#FFFFFF", size=12),
        ),
    )
    fig.update_xaxes(gridcolor=PALETTE.grid_line, zerolinecolor=PALETTE.grid_line)
    fig.update_yaxes(gridcolor=PALETTE.grid_line, zerolinecolor=PALETTE.grid_line)
    return fig


def fig_engine_benchmark(bench: BenchmarkSuiteResults) -> go.Figure:
    """Visual comparison of Persistence & Query Engines latency, speedup, and storage compression."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Latência Mediana Total nas 3 Cargas (ms — Menor é Melhor)",
            "Pegada de Armazenamento em Disco (MB — 122.200 Registros Reais)",
        ),
        horizontal_spacing=0.14,
    )

    eng = bench.engine_summary.sort_values("latencia_total_mediana_ms", ascending=True)
    bar_colors = [
        PALETTE.vibrant_cyan if i == 0 else PALETTE.deep_teal if i == 1 else PALETTE.orbital_blue
        for i in range(len(eng))
    ]

    fig.add_trace(
        go.Bar(
            y=eng["engine"],
            x=eng["latencia_total_mediana_ms"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#FFFFFF", width=0.6)),
            text=[
                f"{ms:.1f} ms ({sp:.1f}x)"
                for ms, sp in zip(
                    eng["latencia_total_mediana_ms"], eng["speedup_vs_sqlite_sem_indice"]
                )
            ],
            textposition="auto",
            name="Latência Mediana (ms)",
            hovertemplate="<b>%{y}</b><br>Latência Total: %{x:.2f} ms<extra></extra>",
        ),
        row=1,
        col=1,
    )

    st = bench.storage_footprint.sort_values("tamanho_mb", ascending=True)
    st_colors = [PALETTE.vibrant_cyan, PALETTE.deep_teal, PALETTE.orbital_blue, PALETTE.space_violet]
    fig.add_trace(
        go.Bar(
            y=st["formato_persistencia"],
            x=st["tamanho_mb"],
            orientation="h",
            marker=dict(color=st_colors[: len(st)], line=dict(color="#FFFFFF", width=0.6)),
            text=[
                f"{mb:.2f} MB ({cr:.2f}x)"
                for mb, cr in zip(st["tamanho_mb"], st["razao_compressao_vs_bruto"])
            ],
            textposition="auto",
            name="Tamanho em Disco (MB)",
            hovertemplate="<b>%{y}</b><br>Tamanho: %{x:.2f} MB<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(showlegend=False)
    return _apply_cosmic_layout(
        fig,
        "Benchmarking de Engenharia de Dados: Motores Analíticos & Persistência Relacional",
        height=440,
    )


def fig_ml_benchmark(bench: BenchmarkSuiteResults) -> go.Figure:
    """Compare Supervised Actuarial Risk Classifiers & Empirical Launch Failure Drivers."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Modelos Atuariais de Sinistro (5-Fold CV em 7.418 Lançamentos)",
            "Importância Empírica dos Fatores de Risco de Lançamento (%)",
        ),
        horizontal_spacing=0.16,
    )

    clf = bench.classification_benchmark.copy()
    fig.add_trace(
        go.Bar(
            name="ROC-AUC (Discriminação)",
            x=clf["modelo_atuarial"],
            y=clf["roc_auc"],
            marker_color=PALETTE.vibrant_cyan,
            text=[f"{v:.3f}" for v in clf["roc_auc"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="1 - Brier Score (Calibração Atuarial)",
            x=clf["modelo_atuarial"],
            y=1.0 - clf["brier_score_calibracao"],
            marker_color=PALETTE.orbital_blue,
            text=[f"{1.0 - v:.3f}" for v in clf["brier_score_calibracao"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    imp = bench.feature_importances.sort_values("importancia_relativa_pct", ascending=True)
    fig.add_trace(
        go.Bar(
            y=imp["fator_risco"],
            x=imp["importancia_relativa_pct"],
            orientation="h",
            marker_color=PALETTE.deep_teal,
            text=[f"{v:.1f}%" for v in imp["importancia_relativa_pct"]],
            textposition="auto",
            name="Importância (%)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(range=[0.65, 1.02], row=1, col=1)
    fig.update_xaxes(tickangle=-18, row=1, col=1)
    fig.update_layout(barmode="group")
    return _apply_cosmic_layout(
        fig,
        "Benchmarking de Mineração Preditiva: Subscrição Atuarial Aeroespacial (Sem Vazamento Temporal)",
        height=480,
    )


def fig_information_trends(trends: TrendAnalysisSummary) -> go.Figure:
    """3-panel historical & projected time-series of Aerospace Information Production (1957-2030)."""
    # Exclude partial current year 2026 from annual rate curves so incomplete year doesn't visually dip,
    # while annotating 2026 YTD & plotting 2026-2030 full-year projection cleanly!
    hist = trends.annual_series[trends.annual_series["ano"] <= 2025].copy()
    fc = trends.forecast_2030.copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "1. Produção Anual de Telemetria & Objetos Catalogados em Órbita (NORAD SATCAT, 1957–2025)",
            "2. Produção Anual de Descobertas Astrométricas de Asteroides Próximos à Terra (NASA JPL SBDB)",
            "3. Índice Composto de Produção de Informações (Base 2010 = 100) + Projeção Log-Linear 2026–2030 (IC 95%)",
        ),
    )

    # Row 1: Payloads vs Debris per year
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
            name="Detritos & Estágios Catalogados (DEB + R/B)",
            line=dict(color=PALETTE.accent_gold, width=1.8, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Row 2: Annual NEO discoveries
    fig.add_trace(
        go.Bar(
            x=hist["ano"],
            y=hist["neos_descobertos_ano"],
            name="NEOs Descobertos / Ano (NASA JPL)",
            marker_color=PALETTE.deep_teal,
        ),
        row=2,
        col=1,
    )

    # Row 3: Composite Information Index + 2026-2030 Forecast with 95% CI band
    fig.add_trace(
        go.Scatter(
            x=hist["ano"],
            y=hist["indice_producao_informacao"],
            mode="lines+markers",
            marker=dict(size=4),
            name="Índice Empírico Observado (1957–2025)",
            line=dict(color=PALETTE.vibrant_cyan, width=2.8),
        ),
        row=3,
        col=1,
    )

    # Confidence band
    fig.add_trace(
        go.Scatter(
            x=list(fc["ano"]) + list(fc["ano"])[::-1],
            y=list(fc["ic95_superior"]) + list(fc["ic95_inferior"])[::-1],
            fill="toself",
            fillcolor="rgba(28, 163, 158, 0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervalo de Predição 95% (2026–2030)",
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
            name="Projeção Log-Linear NewSpace (2026–2030)",
            line=dict(color=PALETTE.accent_mint, width=2.6, dash="dash"),
            marker=dict(size=7, symbol="diamond"),
        ),
        row=3,
        col=1,
    )

    # Structural NewSpace inflection vertical line (2015: First orbital booster landing & reusable era)
    for r in (1, 2, 3):
        fig.add_vline(
            x=2015,
            line_dash="dash",
            line_color="#A78BFA",
            line_width=1.4,
            row=r,
            col=1,
        )

    fig.add_annotation(
        x=2015,
        y=float(hist["cargas_uteis_pay_ano"].max()) * 0.88,
        text="Inflexão NewSpace (2015)<br>Reusabilidade & Megaconstelações",
        showarrow=True,
        arrowhead=2,
        ax=-85,
        ay=-15,
        font=dict(color="#E9D5FF", size=11),
        bgcolor="rgba(35, 20, 91, 0.85)",
        bordercolor="#A78BFA",
        row=1,
        col=1,
    )

    return _apply_cosmic_layout(
        fig,
        "Tendência Empírica de Produção de Informações Espaciais e Projeção 2030",
        height=760,
    )


def fig_observability_and_corridors(
    datasets: MinedDatasets, trends: TrendAnalysisSummary
) -> go.Figure:
    """Dual diagnostic: (1) Information Volume vs Orbit Uncertainty OCC; (2) Orbital Corridor Congestion."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Lei Informação × Risco: Mediana de Observações por Incerteza OCC",
            "Tráfego Real nos Corredores Orbitais: Satélites Ativos vs Detritos",
        ),
        horizontal_spacing=0.14,
    )

    occ = trends.observability_vs_uncertainty
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
            name="Mediana de Observações Astrométricas",
            hovertemplate="<b>%{x}</b><br>Mediana de Observações: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    cor = datasets.corridors.copy()
    fig.add_trace(
        go.Bar(
            x=cor["id_corredor"],
            y=cor["satelites_ativos"],
            name="Satélites Ativos Manobráveis",
            marker_color=PALETTE.vibrant_cyan,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=cor["id_corredor"],
            y=cor["detritos_e_estagios_orbita"],
            name="Detritos & Estágios Inertes (DEB + R/B)",
            marker_color=PALETTE.accent_coral,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(tickangle=-25, row=1, col=2)
    fig.update_layout(barmode="stack")
    return _apply_cosmic_layout(
        fig,
        "Qualidade Informacional Astrométrica & Bifurcação de Detritos nos Corredores Orbitais",
        height=460,
    )


def fig_topsis_prospecting_map(selection: SelectionDecisionOutput) -> go.Figure:
    """Astrodynamic Trade-Off Scatter of qualified NASA NHATS targets colored by TOPSIS score."""
    cand = selection.ranked_asteroids.head(250).copy()
    if cand.empty:
        return _apply_cosmic_layout(go.Figure(), "Nenhum Asteroide Atendeu aos Filtros Selecionados", 420)

    sizes = np.clip(np.log1p(cand["diametro_fisico_ref_m"]) * 4.2, 8.0, 32.0)

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
                    cand["classe_espectral"],
                    np.round(cand["diametro_fisico_ref_m"], 1),
                    cand["num_observacoes"],
                    cand["codigo_incerteza_occ"],
                    cand["score_topsis"],
                ]
            ),
            hovertemplate=(
                "<b>#%{customdata[0]} — %{customdata[1]}</b><br>"
                "Δv Mínimo: %{x:.3f} km/s | Duração: %{y:.0f} dias<br>"
                "Classe Espectral: %{customdata[2]} | Diâmetro Ref: %{customdata[3]} m<br>"
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
    return _apply_cosmic_layout(
        fig,
        f"Mapa de Decisão Multicritério TOPSIS — Top {len(cand)} Alvos Qualificados (NASA JPL NHATS)",
        height=500,
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
                name=f"Rating Atuarial {rating}",
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
        height=460,
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
                cols_show = [
                    "ranking_topsis",
                    "pdes",
                    "nome_completo",
                    "classe_espectral",
                    "min_delta_v_kms",
                    "duracao_min_dv_dias",
                    "diametro_fisico_ref_m",
                    "num_observacoes",
                    "codigo_incerteza_occ",
                    "score_topsis",
                ]
                display(res.ranked_asteroids[cols_show].head(12))

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
    """Export interactive, client-side customizable standalone HTML5 cockpit (zero backend needed)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "newspace_dashboard.html"

    # Serialize static figures to Plotly JSON specs
    f_eng_json = fig_engine_benchmark(bench).to_json()
    f_ml_json = fig_ml_benchmark(bench).to_json()
    f_tr_json = fig_information_trends(trends).to_json()
    f_ob_json = fig_observability_and_corridors(datasets, trends).to_json()
    f_ac_json = fig_actuarial_scenarios_matrix(selection).to_json()

    # Prepare compact JSON of all 7,169 real NASA NHATS targets for instant client-side TOPSIS customization!
    nhats_all = datasets.asteroids[datasets.asteroids["acessivel_nhats"] == 1].copy()
    nhats_all = nhats_all.dropna(
        subset=["min_delta_v_kms", "duracao_min_dv_dias", "diametro_fisico_ref_m"]
    ).copy()

    client_records = []
    for _, r in nhats_all.iterrows():
        client_records.append(
            {
                "spkid": int(r["spkid"]),
                "pdes": str(r["pdes"]),
                "name": str(r["nome_completo"]),
                "spec": str(r["classe_espectral"]),
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

    qual_lv_records = selection.qualified_launchers.head(5)[
        ["modelo_veiculo", "agencia_principal", "total_lancamentos", "taxa_sucesso_pct", "wilson_lower_95_pct"]
    ].to_dict(orient="records")

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
    background: linear-gradient(90deg, var(--bg) 0%, var(--violet) 45%, var(--blue) 100%);
    border-bottom: 2px solid var(--cyan);
    padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
  }}
  .brand-title {{ font-size: 21px; font-weight: 800; letter-spacing: 0.02em; }}
  .brand-sub {{ font-size: 12px; color: #cbd5e1; margin-top: 3px; }}
  .palette-pills {{ display: flex; gap: 6px; align-items: center; font-size: 11px; color: #cbd5e1; }}
  .swatch {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.4); }}
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
  .kpi .val {{ font-size: 24px; font-weight: 800; color: var(--cyan); margin-top: 4px; }}
  .kpi .hint {{ font-size: 11px; color: #cbd5e1; margin-top: 3px; }}
  .chart-card {{
    background: #1D1226; border: 1px solid rgba(28,163,158,0.28); border-radius: 12px;
    padding: 12px; margin-bottom: 20px; box-shadow: 0 10px 26px rgba(0,0,0,0.35);
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
  table.ranking-tbl {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }}
  table.ranking-tbl th {{ background: var(--blue); color: #fff; padding: 9px 10px; text-align: left; position: sticky; top: 0; }}
  table.ranking-tbl td {{ padding: 8px 10px; border-bottom: 1px solid rgba(148,163,184,0.16); }}
  table.ranking-tbl tr:hover {{ background: rgba(28,163,158,0.14); }}
</style>
</head>
<body>
<header>
  <div>
    <div class="brand-title">NEWSPACE DECISION COCKPIT • MINERAÇÃO DE DADOS &amp; RISCO ATUARIAL</div>
    <div class="brand-sub">Dados 100% Reais Auditados (NASA JPL SBDB &amp; NHATS • USSPACECOM NORAD SATCAT • Harvard-Smithsonian GCAT)</div>
  </div>
  <div class="palette-pills">
    <span>Identidade Visual NewSpace:</span>
    <span class="swatch" style="background:#160D18" title="#160D18"></span>
    <span class="swatch" style="background:#23145B" title="#23145B"></span>
    <span class="swatch" style="background:#09456C" title="#09456C"></span>
    <span class="swatch" style="background:#026F6E" title="#026F6E"></span>
    <span class="swatch" style="background:#1CA39E" title="#1CA39E"></span>
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
        <div class="chart-card" style="max-height:430px; overflow-y:auto;">
          <h3 style="margin:6px 8px 10px; color:var(--cyan); font-size:15px;">Top 20 Asteroides Próximos à Terra Ranqueados em Tempo Real (NASA JPL NHATS)</h3>
          <table class="ranking-tbl">
            <thead>
              <tr>
                <th>Rank</th><th>Designação</th><th>Nome Oficial NASA</th><th>Espectro</th>
                <th>Δv (km/s)</th><th>Duração (d)</th><th>Diâm. Ref (m)</th><th>Obs.</th><th>OCC</th><th>Score TOPSIS</th>
              </tr>
            </thead>
            <tbody id="tbl-sim-body"></tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 2: INFORMATION PRODUCTION TRENDS -->
  <section id="tab-trends" class="tab-panel">
    <div class="chart-card"><div id="plot-trends"></div></div>
    <div class="chart-card"><div id="plot-obs-corridors"></div></div>
  </section>

  <!-- TAB 3: DATA MINING BENCHMARK -->
  <section id="tab-bench" class="tab-panel">
    <div class="kpi-row">
      <div class="kpi"><div class="lbl">Motor Analítico Mais Rápido</div><div class="val" style="font-size:19px;">{best_eng['engine']}</div><div class="hint">Latência total: {best_eng['latencia_total_mediana_ms']:.2f} ms ({best_eng['speedup_vs_sqlite_sem_indice']:.1f}x speedup)</div></div>
      <div class="kpi"><div class="lbl">Melhor Clusterização NHATS</div><div class="val" style="font-size:19px;">{bench.clustering_benchmark.iloc[0]['algoritmo']}</div><div class="hint">Silhouette: {bench.clustering_benchmark.iloc[0]['silhouette_score']:.4f}</div></div>
      <div class="kpi"><div class="lbl">Melhor Classificador Atuarial</div><div class="val" style="font-size:19px;">{bench.classification_benchmark.iloc[0]['modelo_atuarial']}</div><div class="hint">ROC-AUC: {bench.classification_benchmark.iloc[0]['roc_auc']:.4f} | Brier: {bench.classification_benchmark.iloc[0]['brier_score_calibracao']:.5f}</div></div>
    </div>
    <div class="chart-card"><div id="plot-eng-bench"></div></div>
    <div class="chart-card"><div id="plot-ml-bench"></div></div>
  </section>

  <!-- TAB 4: ORBITAL CORRIDORS & ACTUARIAL MATRIX -->
  <section id="tab-actuarial" class="tab-panel">
    <div class="chart-card"><div id="plot-actuarial-mat"></div></div>
  </section>
</main>

<script>
const NHATS_DATA = {json.dumps(client_records, ensure_ascii=False)};
const QUAL_LVS = {json.dumps(qual_lv_records, ensure_ascii=False)};
const SPEC_ENG = {f_eng_json};
const SPEC_ML = {f_ml_json};
const SPEC_TR = {f_tr_json};
const SPEC_OB = {f_ob_json};
const SPEC_AC = {f_ac_json};

let currentRanked = [];

function renderStaticPlots() {{
  const cfg = {{responsive: true, displaylogo: false}};
  Plotly.newPlot('plot-eng-bench', SPEC_ENG.data, SPEC_ENG.layout, cfg);
  Plotly.newPlot('plot-ml-bench', SPEC_ML.data, SPEC_ML.layout, cfg);
  Plotly.newPlot('plot-trends', SPEC_TR.data, SPEC_TR.layout, cfg);
  Plotly.newPlot('plot-obs-corridors', SPEC_OB.data, SPEC_OB.layout, cfg);
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
     (grpFilter === 'KNOWN' && d.spec !== 'NAO_ESPECTROGRAFADO') ||
     d.grp.startsWith(grpFilter))
  );

  document.getElementById('kpi-cand-count').textContent = cand.length.toLocaleString('pt-BR');
  document.getElementById('kpi-retention').textContent =
    ((100.0 * cand.length / NHATS_DATA.length).toFixed(2)) + '% dos 7.169 alvos NHATS';

  const tbody = document.getElementById('tbl-sim-body');
  if (cand.length === 0) {{
    document.getElementById('kpi-top1-name').textContent = 'Nenhum candidato';
    document.getElementById('kpi-top1-meta').textContent = 'Relaxe os filtros ao lado';
    tbody.innerHTML = '<tr><td colspan="10">Nenhum asteroide atende simultaneamente às restrições atuais.</td></tr>';
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
      0.10 * (d.spec !== 'NAO_ESPECTROGRAFADO' ? 1.0 : 0.0) +
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
    `Δv ${{top1.dv.toFixed(3)}} km/s • TOPSIS ${{top1.topsis.toFixed(1)}} • ${{top1.spec}}`;

  const topPlot = feats.slice(0, 300);
  const trace = {{
    x: topPlot.map(r => r.dv),
    y: topPlot.map(r => r.dur),
    mode: 'markers',
    type: 'scatter',
    text: topPlot.map((r, idx) =>
      `<b>#${{idx+1}} ${{r.name}}</b><br>Espectro: ${{r.spec}}<br>Diâmetro Ref: ${{r.diam}} m<br>Observações: ${{r.obs}} (OCC ${{r.occ}})<br>TOPSIS: ${{r.topsis.toFixed(2)}}`
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
    title: {{text: `<b>Espaço de Decisão Astrodinâmico — Top ${{topPlot.length}} Alvos Qualificados</b>`, font: {{color: '#F1F5F9', size: 16}}}},
    paper_bgcolor: '#1D1226', plot_bgcolor: '#160D18',
    font: {{color: '#F1F5F9', family: 'Segoe UI, sans-serif'}},
    margin: {{l: 55, r: 30, t: 52, b: 48}},
    xaxis: {{title: 'Δv Mínimo NASA NHATS (km/s)', gridcolor: '#2A2438'}},
    yaxis: {{title: 'Duração da Missão (Dias)', gridcolor: '#2A2438'}}
  }};
  Plotly.react('plot-sim-scatter', [trace], layout, {{responsive: true, displaylogo: false}});

  tbody.innerHTML = feats.slice(0, 20).map((r, idx) => `
    <tr>
      <td><strong>#${{idx + 1}}</strong></td>
      <td>${{r.pdes}}</td>
      <td>${{r.name}}</td>
      <td>${{r.spec}}</td>
      <td style="color:#34D399;font-weight:700;">${{r.dv.toFixed(3)}}</td>
      <td>${{r.dur}}</td>
      <td>${{r.diam.toFixed(1)}}</td>
      <td>${{r.obs}}</td>
      <td>${{r.occ}}</td>
      <td style="color:#1CA39E;font-weight:800;">${{r.topsis.toFixed(2)}}</td>
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
  const header = ['rank','spkid','pdes','nome_oficial','classe_espectral','delta_v_kms','duracao_dias','diametro_ref_m','observacoes','occ','score_topsis'];
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
    return out_path
