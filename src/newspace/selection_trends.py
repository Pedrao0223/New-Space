"""Domain Data Selection Criteria, Information Production Trend Analytics & TOPSIS Actuarial Engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from newspace.config import DEFAULT_CRITERIA
from newspace.ingestion import MinedDatasets


@dataclass
class TrendAnalysisSummary:
    """Empirical time-series and structural regime metrics on information production."""

    annual_series: pd.DataFrame
    era_comparison: pd.DataFrame
    forecast_2030: pd.DataFrame
    observability_vs_uncertainty: pd.DataFrame
    characterization_gap: dict[str, float]
    spearman_obs_vs_occ: tuple[float, float]


@dataclass
class SelectionDecisionOutput:
    """Filtered candidates, TOPSIS ranking, qualified launchers, and actuarial scenarios."""

    funnel_stages: pd.DataFrame
    ranked_asteroids: pd.DataFrame
    qualified_launchers: pd.DataFrame
    actuarial_scenarios: pd.DataFrame


def _classify_era(year: int) -> str:
    if year <= 1990:
        return "1. Era Estatal Clássica (1957–1990)"
    if year <= 2014:
        return "2. Transição Comercial (1991–2014)"
    return "3. Era NewSpace Exponencial (2015–2026)"


def _compute_cagr(start_val: float, end_val: float, n_years: int) -> float:
    if start_val <= 0 or end_val <= 0 or n_years <= 0:
        return 0.0
    return round(100.0 * ((end_val / start_val) ** (1.0 / n_years) - 1.0), 2)


def analyze_information_production_trends(
    datasets: MinedDatasets,
    start_year: int = 1957,
    end_year: int = 2026,
) -> TrendAnalysisSummary:
    """Quantify historical and NewSpace exponential growth in aerospace information production."""
    years = pd.DataFrame({"ano": list(range(start_year, end_year + 1))})

    # 1. Orbital Tracking & Telemetry Objects per Year (NORAD SATCAT)
    sat = datasets.satellites.dropna(subset=["ano_lancamento"]).copy()
    sat["ano"] = sat["ano_lancamento"].astype(int)
    sat_yr = (
        sat[(sat["ano"] >= start_year) & (sat["ano"] <= end_year)]
        .groupby("ano", as_index=False)
        .agg(
            objetos_rastreados_ano=("norad_id", "count"),
            cargas_uteis_pay_ano=("tipo_objeto", lambda s: int((s == "PAY").sum())),
            detritos_estagios_ano=("tipo_objeto", lambda s: int(s.isin(["DEB", "R/B"]).sum())),
            satelites_ativos_atuais_safra=("ativo", "sum"),
        )
    )

    # 2. Asteroid Discovery & Astrometric Observation Volume per Discovery Cohort Year (NASA SBDB)
    ast = datasets.asteroids.dropna(subset=["primeira_obs_ano"]).copy()
    ast["ano"] = ast["primeira_obs_ano"].astype(int)
    ast_yr = (
        ast[(ast["ano"] >= start_year) & (ast["ano"] <= end_year)]
        .groupby("ano", as_index=False)
        .agg(
            neos_descobertos_ano=("spkid", "count"),
            alvos_nhats_descobertos_ano=("acessivel_nhats", "sum"),
            observacoes_astrometricas_geradas=("num_observacoes", "sum"),
            incerteza_occ_mediana=("codigo_incerteza_occ", "median"),
        )
    )

    # 3. Rocket Launch Cadence, Payload Deployment & Empirical Reliability (GCAT)
    lan = datasets.launches.copy()
    lan_yr = (
        lan[(lan["ano_lancamento"] >= start_year) & (lan["ano_lancamento"] <= end_year)]
        .rename(columns={"ano_lancamento": "ano"})
        .groupby("ano", as_index=False)
        .agg(
            lancamentos_orbitais_ano=("launch_tag", "count"),
            sucessos_orbitais_ano=("sucesso_pleno", "sum"),
            cargas_lancadas_gcat_ano=("qtd_cargas", "sum"),
            missoes_espaco_profundo_ano=("espaco_profundo", "sum"),
        )
    )

    annual = (
        years.merge(sat_yr, on="ano", how="left")
        .merge(ast_yr, on="ano", how="left")
        .merge(lan_yr, on="ano", how="left")
    )
    fill_zero_cols = [
        "objetos_rastreados_ano",
        "cargas_uteis_pay_ano",
        "detritos_estagios_ano",
        "satelites_ativos_atuais_safra",
        "neos_descobertos_ano",
        "alvos_nhats_descobertos_ano",
        "observacoes_astrometricas_geradas",
        "lancamentos_orbitais_ano",
        "sucessos_orbitais_ano",
        "cargas_lancadas_gcat_ano",
        "missoes_espaco_profundo_ano",
    ]
    for col in fill_zero_cols:
        annual[col] = annual[col].fillna(0).astype(int)

    annual["taxa_sucesso_lancamento_pct"] = np.where(
        annual["lancamentos_orbitais_ano"] > 0,
        np.round(100.0 * annual["sucessos_orbitais_ano"] / annual["lancamentos_orbitais_ano"], 2),
        np.nan,
    )
    annual["cargas_por_lancamento"] = np.where(
        annual["lancamentos_orbitais_ano"] > 0,
        np.round(annual["cargas_lancadas_gcat_ano"] / annual["lancamentos_orbitais_ano"], 2),
        0.0,
    )
    annual["catalogo_orbital_acumulado"] = annual["objetos_rastreados_ano"].cumsum()
    annual["catalogo_neos_acumulado"] = annual["neos_descobertos_ano"].cumsum()
    annual["era_espacial"] = annual["ano"].map(_classify_era)

    # Composite Normalized Information Production Index (Base 2010 = 100) across the 3 telemetry streams
    # Combines orbital objects tracked + NEO discoveries + astrometric measurements
    raw_signal = (
        0.45 * (annual["objetos_rastreados_ano"] / max(annual["objetos_rastreados_ano"].mean(), 1.0))
        + 0.35 * (annual["neos_descobertos_ano"] / max(annual["neos_descobertos_ano"].mean(), 1.0))
        + 0.20 * (annual["cargas_lancadas_gcat_ano"] / max(annual["cargas_lancadas_gcat_ano"].mean(), 1.0))
    )
    ref_2010 = float(raw_signal.loc[annual["ano"] == 2010].iloc[0]) if (annual["ano"] == 2010).any() else 1.0
    annual["indice_producao_informacao"] = np.round(100.0 * raw_signal / max(ref_2010, 0.01), 2)

    # Structural Regime Comparison (excluding partial current year 2026 only for annual averages if needed,
    # or reporting exact full-year averages up to 2025 vs total volume)
    full_years = annual[annual["ano"] <= 2025].copy()
    era_comp = (
        full_years.groupby("era_espacial", as_index=False)
        .agg(
            anos_completos=("ano", "count"),
            media_anual_obj_orbitais=("objetos_rastreados_ano", "mean"),
            media_anual_cargas_pay=("cargas_uteis_pay_ano", "mean"),
            media_anual_neos_descobertos=("neos_descobertos_ano", "mean"),
            media_cargas_por_foguete=("cargas_por_lancamento", "mean"),
            taxa_sucesso_media_pct=("taxa_sucesso_lancamento_pct", "mean"),
            indice_info_medio=("indice_producao_informacao", "mean"),
        )
        .round(2)
    )

    # Log-linear regression projection 2026-2030 trained on complete NewSpace years (2015-2025)
    train_ns = annual[(annual["ano"] >= 2015) & (annual["ano"] <= 2025)].copy()
    x_tr = train_ns["ano"].to_numpy(dtype=np.float64)
    forecast_rows = []
    future_years = np.arange(2026, 2031, dtype=np.float64)

    # Fit log-linear trend on Composite Information Index & Annual NEO Discoveries & Orbital Objects
    slope_idx, intercept_idx, r_val, _, std_err = stats.linregress(
        x_tr, np.log(np.maximum(train_ns["indice_producao_informacao"].to_numpy(), 1.0))
    )
    slope_sat, intercept_sat, _, _, _ = stats.linregress(
        x_tr, np.log(np.maximum(train_ns["objetos_rastreados_ano"].to_numpy(), 1.0))
    )
    slope_neo, intercept_neo, _, _, _ = stats.linregress(
        x_tr, np.log(np.maximum(train_ns["neos_descobertos_ano"].to_numpy(), 1.0))
    )

    resid_std = float(
        np.std(
            np.log(np.maximum(train_ns["indice_producao_informacao"].to_numpy(), 1.0))
            - (intercept_idx + slope_idx * x_tr),
            ddof=2,
        )
    )

    for fy in future_years:
        log_pred = intercept_idx + slope_idx * fy
        pred_idx = float(np.exp(log_pred))
        ci_low = float(np.exp(log_pred - 1.96 * resid_std))
        ci_high = float(np.exp(log_pred + 1.96 * resid_std))
        pred_sat = int(round(np.exp(intercept_sat + slope_sat * fy)))
        pred_neo = int(round(np.exp(intercept_neo + slope_neo * fy)))
        forecast_rows.append(
            {
                "ano": int(fy),
                "indice_info_projetado": round(pred_idx, 1),
                "ic95_inferior": round(ci_low, 1),
                "ic95_superior": round(ci_high, 1),
                "objetos_orbitais_ano_proj": pred_sat,
                "neos_descobertos_ano_proj": pred_neo,
                "r2_ajuste_loglinear": round(float(r_val**2), 4),
                "cagr_newspace_2015_2025_pct": _compute_cagr(
                    float(train_ns["indice_producao_informacao"].iloc[0]),
                    float(train_ns["indice_producao_informacao"].iloc[-1]),
                    10,
                ),
            }
        )
    forecast_df = pd.DataFrame(forecast_rows)

    # Empirical Information-to-Risk Law: Observability vs Orbit Uncertainty (OCC)
    valid_occ = datasets.asteroids.dropna(subset=["codigo_incerteza_occ", "num_observacoes"]).copy()
    valid_occ["codigo_incerteza_occ_int"] = valid_occ["codigo_incerteza_occ"].astype(int).clip(0, 9)
    rho, pval = stats.spearmanr(
        valid_occ["num_observacoes"].to_numpy(),
        valid_occ["codigo_incerteza_occ_int"].to_numpy(),
    )
    obs_vs_occ = (
        valid_occ.groupby("codigo_incerteza_occ_int", as_index=False)
        .agg(
            quantidade_asteroides=("spkid", "count"),
            mediana_observacoes=("num_observacoes", "median"),
            mediana_arco_dias=("arco_observacao_dias", "median"),
            pct_acessivel_nhats=("acessivel_nhats", lambda s: round(100.0 * float(s.mean()), 2)),
        )
        .sort_values("codigo_incerteza_occ_int")
    )

    # Characterization gap metrics (crucial strategic insight from real NASA data)
    total_neos = len(datasets.asteroids)
    spec_known = int((datasets.asteroids["classe_espectral"] != "NAO_ESPECTROGRAFADO").sum())
    diam_known = int(datasets.asteroids["diametro_medido_km"].notna().sum())
    nhats_sub = datasets.asteroids[datasets.asteroids["acessivel_nhats"] == 1]
    nhats_spec_known = int((nhats_sub["classe_espectral"] != "NAO_ESPECTROGRAFADO").sum())

    gap_stats = {
        "total_neos_catalogados": total_neos,
        "neos_com_espectro_real": spec_known,
        "pct_neos_com_espectro": round(100.0 * spec_known / max(total_neos, 1), 2),
        "neos_com_diametro_medido": diam_known,
        "pct_neos_com_diametro": round(100.0 * diam_known / max(total_neos, 1), 2),
        "total_nhats_acessiveis": len(nhats_sub),
        "nhats_com_espectro_real": nhats_spec_known,
        "pct_nhats_com_espectro": round(100.0 * nhats_spec_known / max(len(nhats_sub), 1), 2),
    }

    return TrendAnalysisSummary(
        annual_series=annual,
        era_comparison=era_comp,
        forecast_2030=forecast_df,
        observability_vs_uncertainty=obs_vs_occ,
        characterization_gap=gap_stats,
        spearman_obs_vs_occ=(round(float(rho), 4), float(pval)),
    )


def apply_selection_criteria_and_topsis(
    datasets: MinedDatasets,
    max_delta_v_kms: float = DEFAULT_CRITERIA.max_delta_v_kms,
    max_duration_days: int = DEFAULT_CRITERIA.max_duration_days,
    max_orbit_uncertainty: int = DEFAULT_CRITERIA.max_orbit_uncertainty,
    min_observations: int = DEFAULT_CRITERIA.min_observations,
    min_diameter_m: float = 25.0,
    only_spectroscopically_characterized: bool = False,
    min_launcher_flights: int = DEFAULT_CRITERIA.min_launcher_flights,
    min_wilson_reliability_pct: float = DEFAULT_CRITERIA.min_wilson_reliability * 100.0,
    only_active_newspace_launchers: bool = True,
    weights: tuple[float, float, float, float] = (0.40, 0.25, 0.20, 0.15),
) -> SelectionDecisionOutput:
    """Apply transparent domain selection gates and rank targets via MCDA TOPSIS + Actuarial Risk."""
    ast = datasets.asteroids.copy()

    # Stage-by-stage selection funnel to audit exact selectivity
    s0 = ast
    s1 = s0[s0["acessivel_nhats"] == 1]
    s2 = s1[
        (s1["min_delta_v_kms"] <= max_delta_v_kms)
        & (s1["duracao_min_dv_dias"] <= max_duration_days)
    ]
    s3 = s2[
        (s2["codigo_incerteza_occ"] <= max_orbit_uncertainty)
        & (s2["num_observacoes"] >= min_observations)
    ]
    s4 = s3[s3["diametro_fisico_ref_m"] >= min_diameter_m]
    if only_spectroscopically_characterized:
        s5 = s4[s4["classe_espectral"] != "NAO_ESPECTROGRAFADO"].copy()
    else:
        s5 = s4.copy()

    funnel_df = pd.DataFrame(
        [
            {"etapa_funil": "1. Catálogo Global NASA SBDB (NEOs)", "candidatos": len(s0)},
            {"etapa_funil": "2. Solução Balística Viável (NASA NHATS)", "candidatos": len(s1)},
            {
                "etapa_funil": f"3. Janela Energética (Δv ≤ {max_delta_v_kms} km/s & Duração ≤ {max_duration_days} d)",
                "candidatos": len(s2),
            },
            {
                "etapa_funil": f"4. Qualidade Informacional (OCC ≤ {max_orbit_uncertainty} & Obs ≥ {min_observations})",
                "candidatos": len(s3),
            },
            {
                "etapa_funil": f"5. Porte Físico Operacional (Diâmetro Ref ≥ {min_diameter_m:.0f} m)",
                "candidatos": len(s4),
            },
            {
                "etapa_funil": "6. Portfólio Qualificado Final (Pós-Filtro Espectral)",
                "candidatos": len(s5),
            },
        ]
    )
    funnel_df["retencao_global_pct"] = np.round(
        100.0 * funnel_df["candidatos"] / max(len(s0), 1), 3
    )

    if s5.empty:
        return SelectionDecisionOutput(
            funnel_stages=funnel_df,
            ranked_asteroids=s5,
            qualified_launchers=pd.DataFrame(),
            actuarial_scenarios=pd.DataFrame(),
        )

    cand = s5.reset_index(drop=True).copy()

    # Composite Information Maturity Index [0, 100] from real astrometric & spectroscopic observables
    log_obs = np.log1p(cand["num_observacoes"].fillna(0.0))
    log_arc = np.log1p(cand["arco_observacao_dias"].fillna(0.0))
    occ_certainty = (9.0 - cand["codigo_incerteza_occ"].clip(0, 9)) / 9.0
    has_spec = (cand["classe_espectral"] != "NAO_ESPECTROGRAFADO").astype(float)
    has_measured_diam = cand["diametro_medido_km"].notna().astype(float)

    obs_norm = log_obs / max(float(log_obs.max()), 1.0)
    arc_norm = log_arc / max(float(log_arc.max()), 1.0)
    cand["indice_maturidade_info"] = np.round(
        100.0 * (0.35 * obs_norm + 0.25 * arc_norm + 0.25 * occ_certainty + 0.10 * has_spec + 0.05 * has_measured_diam),
        2,
    )

    # TOPSIS Matrix Construction:
    # C1 (Benefit): Astrodynamic Efficiency = 1 / (min_delta_v_kms * sqrt(duracao_min_dv_dias))
    # C2 (Benefit): Information Maturity Index
    # C3 (Benefit): Log Physical Scale = log1p(diametro_fisico_ref_m)
    # C4 (Benefit): Trajectory Window Redundancy = log1p(trajetorias_viaveis)
    c1 = 1.0 / (cand["min_delta_v_kms"] * np.sqrt(np.maximum(cand["duracao_min_dv_dias"], 1.0)))
    c2 = cand["indice_maturidade_info"]
    c3 = np.log1p(cand["diametro_fisico_ref_m"].fillna(1.0))
    c4 = np.log1p(cand["trajetorias_viaveis"].fillna(1.0))

    mat = np.column_stack([c1.to_numpy(), c2.to_numpy(), c3.to_numpy(), c4.to_numpy()])
    norms = np.linalg.norm(mat, axis=0)
    norms = np.where(norms == 0, 1.0, norms)
    norm_mat = mat / norms

    w_arr = np.array(weights, dtype=np.float64)
    w_arr = w_arr / max(w_arr.sum(), 1e-9)
    weighted = norm_mat * w_arr

    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.linalg.norm(weighted - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denom = dist_best + dist_worst
    topsis = np.where(denom > 0, 100.0 * dist_worst / denom, 50.0)

    cand["score_topsis"] = np.round(topsis, 2)
    cand = cand.sort_values(
        ["score_topsis", "min_delta_v_kms"], ascending=[False, True]
    ).reset_index(drop=True)
    cand["ranking_topsis"] = np.arange(1, len(cand) + 1)

    # Select Qualified Launch Vehicles via Actuarial Criteria
    lv = datasets.launchers.copy()
    lv_cond = (lv["total_lancamentos"] >= min_launcher_flights) & (
        lv["wilson_lower_95_pct"] >= min_wilson_reliability_pct
    )
    if only_active_newspace_launchers:
        lv_cond = lv_cond & (lv["ativo_era_newspace"] == 1)
    qual_lv = (
        lv[lv_cond]
        .sort_values(["wilson_lower_95_pct", "total_lancamentos"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # Build Integrated Actuarial Viability Scenarios (Top 25 Asteroids x Top 3 Qualified Launchers)
    cislunar_row = datasets.corridors[datasets.corridors["id_corredor"] == "HEO_CISLUNAR"].iloc[0]
    corridor_congestion = float(cislunar_row["indice_congestao_orbital"])

    scenario_rows = []
    top_targets = cand.head(25)
    top_lvs = qual_lv.head(3)

    idx_counter = 1
    for _, t_row in top_targets.iterrows():
        for _, l_row in top_lvs.iterrows():
            # Actuarial Combined Risk Score [0, 100] (lower = safer) grounded on real empirical risk factors:
            # 1) Launcher Failure Risk (Upper 95% failure bound = 100 - wilson_lower_95_pct) -> scaled
            # 2) Orbital Corridor Debris/Congestion Exposure (HEO_CISLUNAR empirical index)
            # 3) Ephemeris & Characterization Uncertainty (100 - indice_maturidade_info)
            # 4) Deep-Space Energy & Duration Exposure (normalized Delta-V & Duration)
            launcher_fail_bound = 100.0 - float(l_row["wilson_lower_95_pct"])
            risk_launcher = min(100.0, launcher_fail_bound * 5.0)
            risk_corridor = corridor_congestion
            risk_ephemeris = 100.0 - float(t_row["indice_maturidade_info"])
            risk_mission_energy = min(
                100.0,
                (float(t_row["min_delta_v_kms"]) / 10.0) * 60.0
                + (float(t_row["duracao_min_dv_dias"]) / 450.0) * 40.0,
            )

            combined_risk = round(
                0.35 * risk_launcher
                + 0.20 * risk_corridor
                + 0.25 * risk_ephemeris
                + 0.20 * risk_mission_energy,
                2,
            )

            if combined_risk <= 22.0:
                rating = "AA"
            elif combined_risk <= 30.0:
                rating = "A"
            elif combined_risk <= 40.0:
                rating = "BB"
            elif combined_risk <= 52.0:
                rating = "B"
            else:
                rating = "C"

            scenario_rows.append(
                {
                    "id_cenario": f"NS_SCEN_{idx_counter:03d}",
                    "spkid": int(t_row["spkid"]),
                    "pdes": t_row["pdes"],
                    "nome_asteroide": t_row["nome_completo"],
                    "classe_espectral": t_row["classe_espectral"],
                    "grupo_mineralogico": t_row["grupo_mineralogico"],
                    "id_veiculo": l_row["id_veiculo"],
                    "modelo_veiculo": l_row["modelo_veiculo"],
                    "agencia_principal": l_row["agencia_principal"],
                    "id_corredor_transito": "HEO_CISLUNAR",
                    "min_delta_v_kms": round(float(t_row["min_delta_v_kms"]), 3),
                    "duracao_dias": round(float(t_row["duracao_min_dv_dias"]), 0),
                    "diametro_fisico_ref_m": round(float(t_row["diametro_fisico_ref_m"]), 1),
                    "num_observacoes": int(t_row["num_observacoes"]),
                    "indice_maturidade_info": float(t_row["indice_maturidade_info"]),
                    "confiabilidade_wilson_pct": float(l_row["wilson_lower_95_pct"]),
                    "score_topsis": float(t_row["score_topsis"]),
                    "score_risco_combinado": combined_risk,
                    "rating_atuarial": rating,
                }
            )
            idx_counter += 1

    scenarios_df = (
        pd.DataFrame(scenario_rows)
        .sort_values(["score_topsis", "score_risco_combinado"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return SelectionDecisionOutput(
        funnel_stages=funnel_df,
        ranked_asteroids=cand,
        qualified_launchers=qual_lv,
        actuarial_scenarios=scenarios_df,
    )
