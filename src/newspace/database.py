"""Relational SQLite persistence layer & analytical Parquet export with Foreign Key integrity."""

from __future__ import annotations

from contextlib import closing
import sqlite3
from pathlib import Path

import pandas as pd

from newspace.config import DATA_PROCESSED_DIR, DB_PATH
from newspace.ingestion import MinedDatasets


SCHEMA_DDL = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS CENARIO_VIABILIDADE_ATUARIAL;
DROP TABLE IF EXISTS HISTORICO_LANCAMENTO;
DROP TABLE IF EXISTS OBJETO_TRAFEGO_ORBITAL;
DROP TABLE IF EXISTS ASTEROIDE_PROSPECCAO;
DROP TABLE IF EXISTS CORREDOR_ORBITAL;
DROP TABLE IF EXISTS VEICULO_LANCADOR;

CREATE TABLE VEICULO_LANCADOR (
    id_veiculo TEXT PRIMARY KEY,
    modelo_veiculo TEXT NOT NULL UNIQUE,
    agencia_principal TEXT NOT NULL,
    pais_estado TEXT NOT NULL,
    total_lancamentos INTEGER NOT NULL,
    sucessos_plenos INTEGER NOT NULL,
    sucessos_parciais INTEGER NOT NULL,
    falhas INTEGER NOT NULL,
    missoes_espaco_profundo INTEGER NOT NULL,
    cargas_totais_implantadas INTEGER NOT NULL,
    taxa_sucesso_pct REAL NOT NULL,
    wilson_lower_95_pct REAL NOT NULL,
    confiabilidade_bayes_pct REAL NOT NULL,
    primeiro_voo_ano INTEGER NOT NULL,
    ultimo_voo_ano INTEGER NOT NULL,
    ativo_era_newspace INTEGER NOT NULL
);

CREATE TABLE HISTORICO_LANCAMENTO (
    launch_tag TEXT PRIMARY KEY,
    id_veiculo TEXT NOT NULL REFERENCES VEICULO_LANCADOR(id_veiculo),
    modelo_veiculo TEXT NOT NULL,
    agencia_operador TEXT NOT NULL,
    pais_estado TEXT NOT NULL,
    data_lancamento TEXT NOT NULL,
    ano_lancamento INTEGER NOT NULL,
    sitio_lancamento TEXT,
    codigo_resultado TEXT NOT NULL,
    sucesso_pleno INTEGER NOT NULL,
    sucesso_parcial INTEGER NOT NULL,
    falha_missao INTEGER NOT NULL,
    espaco_profundo INTEGER NOT NULL,
    qtd_cargas INTEGER NOT NULL
);

CREATE TABLE CORREDOR_ORBITAL (
    id_corredor TEXT PRIMARY KEY,
    nome_corredor TEXT NOT NULL,
    regime_orbital TEXT NOT NULL,
    objetos_catalogados_total INTEGER NOT NULL,
    objetos_em_orbita INTEGER NOT NULL,
    satelites_ativos INTEGER NOT NULL,
    detritos_e_estagios_orbita INTEGER NOT NULL,
    razao_detritos_pct REAL NOT NULL,
    expansao_pos_2021_pct REAL NOT NULL,
    apogeu_mediano_km REAL,
    inclinacao_mediana_graus REAL,
    indice_congestao_orbital REAL NOT NULL
);

CREATE TABLE OBJETO_TRAFEGO_ORBITAL (
    norad_id INTEGER PRIMARY KEY,
    id_corredor TEXT NOT NULL REFERENCES CORREDOR_ORBITAL(id_corredor),
    object_id TEXT,
    nome_objeto TEXT NOT NULL,
    tipo_objeto TEXT NOT NULL,
    status_operacional TEXT,
    em_orbita INTEGER NOT NULL,
    ativo INTEGER NOT NULL,
    eh_detrito_ou_estagio INTEGER NOT NULL,
    operador_pais TEXT,
    sitio_lancamento TEXT,
    data_lancamento TEXT,
    ano_lancamento INTEGER,
    data_decaimento TEXT,
    periodo_min REAL,
    inclinacao_graus REAL,
    apogeu_km REAL,
    perigeu_km REAL,
    rcs_m2 REAL,
    centro_orbital TEXT
);

CREATE TABLE ASTEROIDE_PROSPECCAO (
    spkid INTEGER PRIMARY KEY,
    pdes TEXT NOT NULL,
    nome_completo TEXT NOT NULL,
    pha TEXT,
    mag_absoluta_h REAL,
    diametro_medido_km REAL,
    diametro_fisico_ref_m REAL,
    albedo REAL,
    classe_espectral TEXT NOT NULL,
    grupo_mineralogico TEXT NOT NULL,
    excentricidade_e REAL,
    semieixo_maior_au REAL,
    perihelio_au REAL,
    inclinacao_graus REAL,
    moid_au REAL,
    periodo_anos REAL,
    arco_observacao_dias REAL,
    num_observacoes REAL,
    codigo_incerteza_occ REAL,
    primeira_obs_data TEXT,
    primeira_obs_ano INTEGER,
    ultima_obs_data TEXT,
    acessivel_nhats INTEGER NOT NULL,
    min_delta_v_kms REAL,
    duracao_min_dv_dias REAL,
    duracao_mais_curta_dias REAL,
    trajetorias_viaveis REAL,
    janela_obs_inicio TEXT,
    janela_obs_fim TEXT
);

CREATE TABLE CENARIO_VIABILIDADE_ATUARIAL (
    id_cenario TEXT PRIMARY KEY,
    spkid INTEGER NOT NULL REFERENCES ASTEROIDE_PROSPECCAO(spkid),
    id_veiculo TEXT NOT NULL REFERENCES VEICULO_LANCADOR(id_veiculo),
    id_corredor_transito TEXT NOT NULL REFERENCES CORREDOR_ORBITAL(id_corredor),
    min_delta_v_kms REAL NOT NULL,
    duracao_dias REAL NOT NULL,
    indice_maturidade_info REAL NOT NULL,
    confiabilidade_wilson_pct REAL NOT NULL,
    score_topsis REAL NOT NULL,
    score_risco_combinado REAL NOT NULL,
    rating_atuarial TEXT NOT NULL
);
"""

INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_hist_lanc_ano_veic ON HISTORICO_LANCAMENTO(ano_lancamento, id_veiculo, sucesso_pleno, qtd_cargas);
CREATE INDEX IF NOT EXISTS idx_traf_orb_ano_cor ON OBJETO_TRAFEGO_ORBITAL(em_orbita, ano_lancamento, id_corredor, ativo, eh_detrito_ou_estagio, apogeu_km);
CREATE INDEX IF NOT EXISTS idx_ast_nhats_dv ON ASTEROIDE_PROSPECCAO(acessivel_nhats, min_delta_v_kms, codigo_incerteza_occ, num_observacoes, grupo_mineralogico, duracao_min_dv_dias);
CREATE INDEX IF NOT EXISTS idx_ast_ano ON ASTEROIDE_PROSPECCAO(primeira_obs_ano);
ANALYZE;
"""


def persist_relational_and_parquet(
    datasets: MinedDatasets,
    db_path: Path = DB_PATH,
    processed_dir: Path = DATA_PROCESSED_DIR,
    create_indexes: bool = True,
) -> dict[str, int]:
    """Persist all real datasets into normalized SQLite RDBMS and analytical Parquet files."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.executescript(SCHEMA_DDL)
        conn.execute("PRAGMA foreign_keys = ON;")

        datasets.launchers.to_sql("VEICULO_LANCADOR", conn, if_exists="append", index=False)
        datasets.launches.to_sql("HISTORICO_LANCAMENTO", conn, if_exists="append", index=False)
        datasets.corridors.to_sql("CORREDOR_ORBITAL", conn, if_exists="append", index=False)
        datasets.satellites.to_sql("OBJETO_TRAFEGO_ORBITAL", conn, if_exists="append", index=False)
        datasets.asteroids.to_sql("ASTEROIDE_PROSPECCAO", conn, if_exists="append", index=False)

        if create_indexes:
            conn.executescript(INDEX_DDL)

        # Verify strict Foreign Key integrity (zero violations!)
        fk_errors = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if fk_errors:
            raise RuntimeError(f"Foreign key integrity check failed: {fk_errors[:5]}")

        counts = {}
        for tbl in [
            "VEICULO_LANCADOR",
            "HISTORICO_LANCAMENTO",
            "CORREDOR_ORBITAL",
            "OBJETO_TRAFEGO_ORBITAL",
            "ASTEROIDE_PROSPECCAO",
        ]:
            counts[tbl] = int(conn.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()[0])
        conn.commit()

    # Export clean Parquet files for OLAP benchmarking (Polars / DuckDB)
    datasets.launchers.to_parquet(processed_dir / "veiculo_lancador.parquet", index=False)
    datasets.launches.to_parquet(processed_dir / "historico_lancamento.parquet", index=False)
    datasets.corridors.to_parquet(processed_dir / "corredor_orbital.parquet", index=False)
    datasets.satellites.to_parquet(processed_dir / "objeto_trafego_orbital.parquet", index=False)
    datasets.asteroids.to_parquet(processed_dir / "asteroide_prospeccao.parquet", index=False)

    return counts


def save_actuarial_scenarios(
    scenarios_df: pd.DataFrame,
    db_path: Path = DB_PATH,
    processed_dir: Path = DATA_PROCESSED_DIR,
) -> int:
    """Persist computed TOPSIS & Actuarial scenarios into SQLite with FK validation."""
    cols = [
        "id_cenario",
        "spkid",
        "id_veiculo",
        "id_corredor_transito",
        "min_delta_v_kms",
        "duracao_dias",
        "indice_maturidade_info",
        "confiabilidade_wilson_pct",
        "score_topsis",
        "score_risco_combinado",
        "rating_atuarial",
    ]
    clean_df = scenarios_df[cols].copy()
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("DELETE FROM CENARIO_VIABILIDADE_ATUARIAL;")
        clean_df.to_sql("CENARIO_VIABILIDADE_ATUARIAL", conn, if_exists="append", index=False)
        fk_errors = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if fk_errors:
            raise RuntimeError(f"Foreign key violation in CENARIO_VIABILIDADE_ATUARIAL: {fk_errors[:5]}")
        conn.commit()
    clean_df.to_parquet(processed_dir / "cenario_viabilidade_atuarial.parquet", index=False)
    return len(clean_df)
