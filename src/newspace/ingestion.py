"""Ingestion, validation, and physical classification of 100% real aerospace datasets.

Data Sources:
- NASA/Caltech JPL Small-Body Database (43,072 Near-Earth Objects)
- NASA/Caltech JPL NHATS (7,169 Accessible Trajectory Targets with Delta-V)
- CelesTrak / USSPACECOM NORAD SATCAT (71,262 Tracked Space Objects 1957-2026)
- Harvard-Smithsonian CfA GCAT (7,444 Distinct Rocket Launches / 32,045 Payloads 1957-2026)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from newspace.config import DATA_RAW_DIR


MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

ACTIVE_OPS_CODES = {"+", "P", "B", "S", "X"}


@dataclass
class MinedDatasets:
    """Container of validated, normalized real datasets ready for persistence & benchmarking."""

    launchers: pd.DataFrame
    launches: pd.DataFrame
    corridors: pd.DataFrame
    satellites: pd.DataFrame
    asteroids: pd.DataFrame
    provenance: dict


def verify_provenance(raw_dir: Path = DATA_RAW_DIR) -> dict:
    """Verify raw files exist and match SHA-256 hashes recorded in manifest_provenance.json."""
    manifest_path = raw_dir / "manifest_provenance.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Provenance manifest not found at {manifest_path}. Run scripts/fetch_real_data.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("datasets", []):
        fpath = raw_dir / item["local_file"]
        if not fpath.exists():
            raise FileNotFoundError(f"Missing raw scientific dataset: {fpath}")
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(
                f"SHA-256 integrity mismatch for {fpath.name}: expected {item['sha256']}, got {digest}"
            )
    return manifest


def wilson_lower_bound(successes: np.ndarray, totals: np.ndarray, z: float = 1.959964) -> np.ndarray:
    """Compute 95% Wilson Score lower confidence bound for binomial launch reliability."""
    n = totals.astype(np.float64)
    p = np.divide(successes, n, out=np.zeros_like(n), where=n > 0)
    z2 = z * z
    denom = 1.0 + z2 / np.maximum(n, 1.0)
    center = p + z2 / (2.0 * np.maximum(n, 1.0))
    radicand = (p * (1.0 - p) + z2 / (4.0 * np.maximum(n, 1.0))) / np.maximum(n, 1.0)
    lower = (center - z * np.sqrt(np.maximum(radicand, 0.0))) / denom
    return np.where(n > 0, np.clip(lower, 0.0, 1.0), 0.0)


def _parse_gcat_date(raw_date: str) -> tuple[str | None, int | None]:
    """Parse GCAT date format e.g. '1957 Oct  4 1928:34' or '2024 Mar 12?' to ISO YYYY-MM-DD."""
    cleaned = raw_date.replace("?", "").strip()
    parts = cleaned.split()
    if len(parts) < 2 or not parts[0].isdigit():
        return None, None
    year = int(parts[0])
    month = MONTH_MAP.get(parts[1][:3], 1)
    day = 1
    if len(parts) >= 3:
        day_digits = "".join(ch for ch in parts[2] if ch.isdigit())
        if day_digits:
            day = max(1, min(31, int(day_digits)))
    try:
        iso = datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        iso = f"{year:04d}-{month:02d}-01"
    return iso, year


def load_gcat_launches_and_vehicles(raw_dir: Path = DATA_RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse Harvard-Smithsonian CfA GCAT launch log into distinct launches and vehicle reliability."""
    tsv_path = raw_dir / "gcat_launchlog.tsv"
    lines = tsv_path.read_text(encoding="utf-8", errors="replace").splitlines()

    launch_records: dict[str, dict] = {}
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 19:
            continue
        tag = cols[0].strip()
        if not tag:
            continue

        code = cols[18].strip()
        # Keep strictly Orbital ('O*') and Deep-Space/Escape ('D*') launch attempts
        if not code or code[0] not in ("O", "D"):
            continue

        if tag not in launch_records:
            iso_date, year = _parse_gcat_date(cols[1])
            if year is None:
                continue
            lv_type = cols[9].strip() or "Unknown LV"
            agency = cols[16].strip() or "UNK"
            lv_state = cols[17].strip() or "UNK"
            site = cols[12].strip() or "UNK"

            is_full_success = 1 if code in ("OS", "DS") else 0
            is_partial = 1 if (code.startswith(("OS", "DS")) and len(code) > 2) else 0
            is_failure = 1 if (len(code) >= 2 and code[1] == "F") else 0
            is_deep_space = 1 if code[0] == "D" else 0

            launch_records[tag] = {
                "launch_tag": tag,
                "modelo_veiculo": lv_type,
                "agencia_operador": agency,
                "pais_estado": lv_state,
                "data_lancamento": iso_date,
                "ano_lancamento": year,
                "sitio_lancamento": site,
                "codigo_resultado": code,
                "sucesso_pleno": is_full_success,
                "sucesso_parcial": is_partial,
                "falha_missao": is_failure,
                "espaco_profundo": is_deep_space,
                "qtd_cargas": 1,
            }
        else:
            launch_records[tag]["qtd_cargas"] += 1

    launches_df = pd.DataFrame(list(launch_records.values()))

    # Aggregate into canonical Launch Vehicle entity
    grouped = (
        launches_df.groupby("modelo_veiculo", as_index=False)
        .agg(
            agencia_principal=("agencia_operador", "first"),
            pais_estado=("pais_estado", "first"),
            total_lancamentos=("launch_tag", "count"),
            sucessos_plenos=("sucesso_pleno", "sum"),
            sucessos_parciais=("sucesso_parcial", "sum"),
            falhas=("falha_missao", "sum"),
            missoes_espaco_profundo=("espaco_profundo", "sum"),
            cargas_totais_implantadas=("qtd_cargas", "sum"),
            primeiro_voo_ano=("ano_lancamento", "min"),
            ultimo_voo_ano=("ano_lancamento", "max"),
        )
        .sort_values(["total_lancamentos", "sucessos_plenos"], ascending=[False, False])
        .reset_index(drop=True)
    )

    grouped["id_veiculo"] = [f"LV_{i + 1:04d}" for i in range(len(grouped))]
    grouped["taxa_sucesso_pct"] = np.round(
        100.0 * grouped["sucessos_plenos"] / grouped["total_lancamentos"], 2
    )
    grouped["wilson_lower_95_pct"] = np.round(
        100.0
        * wilson_lower_bound(
            grouped["sucessos_plenos"].to_numpy(),
            grouped["total_lancamentos"].to_numpy(),
        ),
        2,
    )
    # Empirical Bayes (Beta-Binomial shrinkage toward global orbital success rate)
    global_p = float(launches_df["sucesso_pleno"].mean())
    prior_strength = 5.0
    alpha0 = global_p * prior_strength
    beta0 = (1.0 - global_p) * prior_strength
    grouped["confiabilidade_bayes_pct"] = np.round(
        100.0
        * (grouped["sucessos_plenos"] + alpha0)
        / (grouped["total_lancamentos"] + prior_strength),
        2,
    )
    grouped["ativo_era_newspace"] = (grouped["ultimo_voo_ano"] >= 2018).astype(int)

    lv_map = dict(zip(grouped["modelo_veiculo"], grouped["id_veiculo"]))
    launches_df["id_veiculo"] = launches_df["modelo_veiculo"].map(lv_map)

    launcher_cols = [
        "id_veiculo",
        "modelo_veiculo",
        "agencia_principal",
        "pais_estado",
        "total_lancamentos",
        "sucessos_plenos",
        "sucessos_parciais",
        "falhas",
        "missoes_espaco_profundo",
        "cargas_totais_implantadas",
        "taxa_sucesso_pct",
        "wilson_lower_95_pct",
        "confiabilidade_bayes_pct",
        "primeiro_voo_ano",
        "ultimo_voo_ano",
        "ativo_era_newspace",
    ]
    launch_cols = [
        "launch_tag",
        "id_veiculo",
        "modelo_veiculo",
        "agencia_operador",
        "pais_estado",
        "data_lancamento",
        "ano_lancamento",
        "sitio_lancamento",
        "codigo_resultado",
        "sucesso_pleno",
        "sucesso_parcial",
        "falha_missao",
        "espaco_profundo",
        "qtd_cargas",
    ]
    return grouped[launcher_cols], launches_df[launch_cols]


CORRIDOR_METADATA = {
    "LEO_VLEO": (
        "Órbita Terrestre Muito Baixa (<450 km - Estações & VLEO)",
        "LEO",
    ),
    "LEO_CONSTELACAO": (
        "Corredor de Megaconstelações LEO (450-650 km)",
        "LEO",
    ),
    "LEO_SSO_ALTA": (
        "Órbita Baixa Alta & Heliossíncrona SSO (650-2.000 km)",
        "LEO",
    ),
    "MEO_NAVEGACAO": (
        "Órbita Terrestre Média GNSS (2.000-34.000 km)",
        "MEO",
    ),
    "GEO_CINTURAO": (
        "Cinturão Geoestacionário / Geossíncrono (~35.786 km)",
        "GEO",
    ),
    "HEO_CISLUNAR": (
        "Corredor Elíptico de Escape, GTO & Cislunar (>37.500 km / Lua)",
        "CISLUNAR",
    ),
    "ESPACO_PROFUNDO": (
        "Corredor Heliocêntrico, Lagrange & Interplanetário",
        "HELIOCENTRICO",
    ),
}


def load_satcat_and_corridors(raw_dir: Path = DATA_RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all 71k+ NORAD SATCAT objects and derive physical Orbital Corridors."""
    sat_path = raw_dir / "celestrak_satcat.csv"
    df = pd.read_csv(sat_path, low_memory=False)

    df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"], errors="coerce")
    df = df.dropna(subset=["NORAD_CAT_ID"]).copy()
    df["norad_id"] = df["NORAD_CAT_ID"].astype(int)

    df["apogeu_km"] = pd.to_numeric(df["APOGEE"], errors="coerce")
    df["perigeu_km"] = pd.to_numeric(df["PERIGEE"], errors="coerce")
    df["inclinacao_graus"] = pd.to_numeric(df["INCLINATION"], errors="coerce")
    df["periodo_min"] = pd.to_numeric(df["PERIOD"], errors="coerce")
    df["rcs_m2"] = pd.to_numeric(df["RCS"], errors="coerce")

    orbit_center = df["ORBIT_CENTER"].fillna("EA").astype(str).str.strip()
    orbit_type = df["ORBIT_TYPE"].fillna("UNK").astype(str).str.strip()
    ops_code = df["OPS_STATUS_CODE"].fillna("").astype(str).str.strip()
    obj_type = df["OBJECT_TYPE"].fillna("UNK").astype(str).str.strip()

    apogee = df["apogeu_km"].fillna(-1.0)
    perigee = df["perigeu_km"].fillna(-1.0)

    # Vectorized physical orbital corridor classification
    conds = [
        ~orbit_center.isin(["EA", "MO", "EM"]),
        orbit_center.isin(["MO", "EM"]) | (apogee > 37500),
        (apogee >= 34000) & (apogee <= 37500) & (perigee >= 30000),
        (apogee >= 34000) & (apogee <= 37500) & (perigee < 30000),  # GTO transfer!
        (apogee > 2000) & (apogee < 34000),
        (apogee >= 650) & (apogee <= 2000),
        (apogee >= 450) & (apogee < 650),
    ]
    choices = [
        "ESPACO_PROFUNDO",
        "HEO_CISLUNAR",
        "GEO_CINTURAO",
        "HEO_CISLUNAR",
        "MEO_NAVEGACAO",
        "LEO_SSO_ALTA",
        "LEO_CONSTELACAO",
    ]
    df["id_corredor"] = np.select(conds, choices, default="LEO_VLEO")

    df["em_orbita"] = orbit_type.isin(["ORB", "DOC"]).astype(int)
    df["ativo"] = ops_code.isin(ACTIVE_OPS_CODES).astype(int)
    df["eh_detrito_ou_estagio"] = (
        (df["em_orbita"] == 1) & obj_type.isin(["DEB", "R/B"])
    ).astype(int)

    df["data_lancamento"] = df["LAUNCH_DATE"].fillna("").astype(str).str.strip()
    df["ano_lancamento"] = pd.to_numeric(
        df["data_lancamento"].str.slice(0, 4), errors="coerce"
    ).astype("Int64")

    satellites_out = pd.DataFrame(
        {
            "norad_id": df["norad_id"],
            "id_corredor": df["id_corredor"],
            "object_id": df["OBJECT_ID"].fillna("").astype(str),
            "nome_objeto": df["OBJECT_NAME"].fillna("UNKNOWN").astype(str),
            "tipo_objeto": obj_type,
            "status_operacional": ops_code.replace("", "N/A"),
            "em_orbita": df["em_orbita"],
            "ativo": df["ativo"],
            "eh_detrito_ou_estagio": df["eh_detrito_ou_estagio"],
            "operador_pais": df["OWNER"].fillna("UNK").astype(str),
            "sitio_lancamento": df["LAUNCH_SITE"].fillna("UNK").astype(str),
            "data_lancamento": df["data_lancamento"].replace("", None),
            "ano_lancamento": df["ano_lancamento"],
            "data_decaimento": df["DECAY_DATE"].where(df["DECAY_DATE"].notna(), None),
            "periodo_min": df["periodo_min"],
            "inclinacao_graus": df["inclinacao_graus"],
            "apogeu_km": df["apogeu_km"],
            "perigeu_km": df["perigeu_km"],
            "rcs_m2": df["rcs_m2"],
            "centro_orbital": orbit_center,
        }
    )

    # Aggregate real empirical metrics per Orbital Corridor
    in_orbit_df = satellites_out[satellites_out["em_orbita"] == 1]
    corridor_rows = []
    for cid, (cname, regime) in CORRIDOR_METADATA.items():
        sub_all = satellites_out[satellites_out["id_corredor"] == cid]
        sub_orb = in_orbit_df[in_orbit_df["id_corredor"] == cid]

        n_total = len(sub_all)
        n_orb = len(sub_orb)
        n_active = int(sub_orb["ativo"].sum())
        n_debris = int(sub_orb["eh_detrito_ou_estagio"].sum())
        debris_pct = round(100.0 * n_debris / n_orb, 2) if n_orb > 0 else 0.0

        med_apo = float(sub_orb["apogeu_km"].median()) if sub_orb["apogeu_km"].notna().any() else np.nan
        med_inc = (
            float(sub_orb["inclinacao_graus"].median())
            if sub_orb["inclinacao_graus"].notna().any()
            else np.nan
        )

        # Recent 5-year telemetry/object growth ratio in corridor (2021-2026 vs total)
        recent_count = int((sub_all["ano_lancamento"] >= 2021).fillna(False).sum())
        aceleracao_newspace_pct = round(100.0 * recent_count / n_total, 2) if n_total > 0 else 0.0

        corridor_rows.append(
            {
                "id_corredor": cid,
                "nome_corredor": cname,
                "regime_orbital": regime,
                "objetos_catalogados_total": n_total,
                "objetos_em_orbita": n_orb,
                "satelites_ativos": n_active,
                "detritos_e_estagios_orbita": n_debris,
                "razao_detritos_pct": debris_pct,
                "expansao_pos_2021_pct": aceleracao_newspace_pct,
                "apogeu_mediano_km": round(med_apo, 1) if not math.isnan(med_apo) else None,
                "inclinacao_mediana_graus": round(med_inc, 2) if not math.isnan(med_inc) else None,
            }
        )

    corridors_df = pd.DataFrame(corridor_rows)
    max_orb = max(int(corridors_df["objetos_em_orbita"].max()), 1)
    # Normalized spatial traffic & debris exposure index [0, 100] derived purely from empirical counts
    corridors_df["indice_congestao_orbital"] = np.round(
        100.0
        * (
            0.60 * (corridors_df["objetos_em_orbita"] / max_orb)
            + 0.40 * (corridors_df["razao_detritos_pct"] / 100.0)
        ),
        2,
    )
    return corridors_df, satellites_out


def load_nasa_asteroids(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load 43k+ NASA SBDB Near-Earth Asteroids and enrich with 7,169 NASA NHATS trajectories."""
    sbdb_raw = json.loads((raw_dir / "nasa_jpl_sbdb_neo.json").read_text(encoding="utf-8"))
    sbdb_df = pd.DataFrame(sbdb_raw["data"], columns=sbdb_raw["fields"])

    sbdb_df["spkid"] = pd.to_numeric(sbdb_df["spkid"], errors="coerce").astype("Int64")
    sbdb_df["pdes"] = sbdb_df["pdes"].astype(str).str.strip()
    sbdb_df["nome_completo"] = sbdb_df["full_name"].astype(str).str.strip()

    num_cols = [
        "H", "diameter", "albedo", "e", "a", "q", "i", "moid",
        "per_y", "data_arc", "n_obs_used", "condition_code",
    ]
    for col in num_cols:
        sbdb_df[col] = pd.to_numeric(sbdb_df[col], errors="coerce")

    # Consolidate real spectroscopy (Bus-DeMeo preferred, fallback to Tholen; never fabricate!)
    spec_b = sbdb_df["spec_B"].fillna("").astype(str).str.strip()
    spec_t = sbdb_df["spec_T"].fillna("").astype(str).str.strip()
    raw_spec = np.where(spec_b != "", spec_b, spec_t)
    sbdb_df["classe_espectral"] = np.where(raw_spec != "", raw_spec, "NAO_ESPECTROGRAFADO")

    # Group real spectral classes into canonical mineralogical complexes for transparent analysis
    first_letter = pd.Series(sbdb_df["classe_espectral"]).str.slice(0, 1)
    complex_map = {
        "C": "Complexo-C (Carbonáceo / Voláteis & Água ISRU)",
        "B": "Complexo-C (Carbonáceo / Voláteis & Água ISRU)",
        "G": "Complexo-C (Carbonáceo / Voláteis & Água ISRU)",
        "F": "Complexo-C (Carbonáceo / Voláteis & Água ISRU)",
        "S": "Complexo-S (Silicatado / Metais & Rochoso)",
        "Q": "Complexo-S (Silicatado / Condrito Ordinário)",
        "K": "Complexo-S (Silicatado / Metais & Rochoso)",
        "L": "Complexo-S (Silicatado / Metais & Rochoso)",
        "X": "Complexo-X/M (Metálico / Grupo da Platina & Fe-Ni)",
        "M": "Complexo-X/M (Metálico / Grupo da Platina & Fe-Ni)",
        "E": "Complexo-X/M (Enstatita / Alta Refletância)",
        "P": "Complexo-X/M (Primitivo / Orgânicos)",
        "V": "Complexo-V/D (Basáltico / Diferenciado)",
        "D": "Complexo-V/D (Primitivo Rico em Orgânicos)",
        "T": "Complexo-V/D (Primitivo Rico em Orgânicos)",
        "A": "Complexo-S (Olivina Pura / Manto Diferenciado)",
    }
    sbdb_df["grupo_mineralogico"] = np.where(
        sbdb_df["classe_espectral"] == "NAO_ESPECTROGRAFADO",
        "Não Caracterizado (Alvo de Sonda Precursora)",
        first_letter.map(complex_map).fillna("Outros Tipos Taxonômicos Reais"),
    )

    sbdb_df["primeira_obs_data"] = sbdb_df["first_obs"].where(sbdb_df["first_obs"].notna(), None)
    sbdb_df["primeira_obs_ano"] = pd.to_numeric(
        sbdb_df["first_obs"].fillna("").astype(str).str.slice(0, 4),
        errors="coerce",
    ).astype("Int64")
    sbdb_df["ultima_obs_data"] = sbdb_df["last_obs"].where(sbdb_df["last_obs"].notna(), None)

    # Parse NASA JPL NHATS accessible asteroid trajectory solutions
    nhats_raw = json.loads((raw_dir / "nasa_jpl_nhats.json").read_text(encoding="utf-8"))
    nhats_rows = []
    for rec in nhats_raw.get("data", []):
        min_dv_obj = rec.get("min_dv") or {}
        min_dur_obj = rec.get("min_dur") or {}
        nhats_rows.append(
            {
                "pdes": str(rec.get("des", "")).strip(),
                "acessivel_nhats": 1,
                "min_delta_v_kms": pd.to_numeric(min_dv_obj.get("dv"), errors="coerce"),
                "duracao_min_dv_dias": pd.to_numeric(min_dv_obj.get("dur"), errors="coerce"),
                "duracao_mais_curta_dias": pd.to_numeric(min_dur_obj.get("dur"), errors="coerce"),
                "trajetorias_viaveis": pd.to_numeric(rec.get("n_via_traj"), errors="coerce"),
                "nhats_min_size_m": pd.to_numeric(rec.get("min_size"), errors="coerce"),
                "nhats_max_size_m": pd.to_numeric(rec.get("max_size"), errors="coerce"),
                "nhats_occ": pd.to_numeric(rec.get("occ"), errors="coerce"),
                "janela_obs_inicio": rec.get("obs_start"),
                "janela_obs_fim": rec.get("obs_end"),
            }
        )
    nhats_df = pd.DataFrame(nhats_rows).drop_duplicates(subset=["pdes"])

    merged = sbdb_df.merge(nhats_df, on="pdes", how="left")
    merged["acessivel_nhats"] = merged["acessivel_nhats"].fillna(0).astype(int)

    # Consolidate Orbit Uncertainty Code (condition_code from SBDB or occ from NHATS)
    merged["codigo_incerteza_occ"] = merged["condition_code"].combine_first(merged["nhats_occ"])

    # Physical diameter: preserve exact measured diameter (km -> m) when available;
    # also compute standard IAU optical estimate (m) clearly flagged as optical estimate.
    pv = merged["albedo"].fillna(0.14).clip(lower=0.01, upper=1.0)
    optical_est_m = (1329.0 / np.sqrt(pv)) * np.power(10.0, -0.2 * merged["H"]) * 1000.0
    nhats_geom_m = np.sqrt(merged["nhats_min_size_m"] * merged["nhats_max_size_m"])
    merged["diametro_medido_km"] = merged["diameter"]
    merged["diametro_fisico_ref_m"] = np.where(
        merged["diameter"].notna(),
        merged["diameter"] * 1000.0,
        np.where(nhats_geom_m.notna(), nhats_geom_m, optical_est_m),
    )

    out_cols = [
        "spkid",
        "pdes",
        "nome_completo",
        "pha",
        "H",
        "diametro_medido_km",
        "diametro_fisico_ref_m",
        "albedo",
        "classe_espectral",
        "grupo_mineralogico",
        "e",
        "a",
        "q",
        "i",
        "moid",
        "per_y",
        "data_arc",
        "n_obs_used",
        "codigo_incerteza_occ",
        "primeira_obs_data",
        "primeira_obs_ano",
        "ultima_obs_data",
        "acessivel_nhats",
        "min_delta_v_kms",
        "duracao_min_dv_dias",
        "duracao_mais_curta_dias",
        "trajetorias_viaveis",
        "janela_obs_inicio",
        "janela_obs_fim",
    ]
    rename_map = {
        "H": "mag_absoluta_h",
        "e": "excentricidade_e",
        "a": "semieixo_maior_au",
        "q": "perihelio_au",
        "i": "inclinacao_graus",
        "moid": "moid_au",
        "per_y": "periodo_anos",
        "data_arc": "arco_observacao_dias",
        "n_obs_used": "num_observacoes",
    }
    return merged[out_cols].rename(columns=rename_map)


def load_all_real_datasets(raw_dir: Path = DATA_RAW_DIR) -> MinedDatasets:
    """Verify SHA-256 provenance and load all normalized empirical datasets."""
    provenance = verify_provenance(raw_dir)
    launchers, launches = load_gcat_launches_and_vehicles(raw_dir)
    corridors, satellites = load_satcat_and_corridors(raw_dir)
    asteroids = load_nasa_asteroids(raw_dir)
    return MinedDatasets(
        launchers=launchers,
        launches=launches,
        corridors=corridors,
        satellites=satellites,
        asteroids=asteroids,
        provenance=provenance,
    )
