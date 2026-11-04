"""Programmatically construct, execute in-process via IPython, and validate the Jupyter Notebook."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
import shutil
from pathlib import Path

from newspace.config import NOTEBOOKS_DIR, PROJECT_ROOT

_IPY_TMP = PROJECT_ROOT / ".jupyter_runtime" / "ipython"
_IPY_TMP.mkdir(parents=True, exist_ok=True)
os.environ["IPYTHONDIR"] = str(_IPY_TMP)

from IPython.core.displaypub import DisplayPublisher
from IPython.core.interactiveshell import InteractiveShell
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output


class NotebookCapturePublisher(DisplayPublisher):
    """Capture rich IPython display() MIME bundles directly into nbformat v4 outputs."""

    def __init__(self, shell=None, *args, **kwargs):
        super().__init__(shell=shell, *args, **kwargs)
        self.outputs: list = []

    def publish(self, data, metadata=None, source=None, *, transient=None, update=False, **kwargs):
        clean_data = {}
        for mime, val in (data or {}).items():
            clean_data[mime] = val
        if clean_data:
            self.outputs.append(
                new_output(
                    output_type="display_data",
                    data=clean_data,
                    metadata=metadata or {},
                )
            )

    def clear_output(self, wait=False):
        self.outputs.clear()


def create_notebook_structure() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (NewSpace Analytics)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.14",
    }

    cells = [
        new_markdown_cell(
            """# 🛰️ NewSpace Analytics: Benchmarking de Mineração de Dados, Tendências de Produção de Informações e Suporte à Decisão

**Domínio de Negócio da Equipe:** *Operações Integradas de Prospecção Mineral de Asteroides Próximos à Terra (NEAs), Gestão de Tráfego em Corredores Orbitais e Subscrição Atuarial Aeroespacial.*

---

### Arquitetura e Governança Científica (Zero Dados Fictícios)
Esta aplicação evolui a modelagem relacional concebida na etapa anterior sobre **100% de bases científicas públicas, oficiais e auditáveis via hash criptográfico SHA-256**, sem qualquer geração sintética ou valores arbitrários:

1. **NASA / Caltech JPL — Small-Body Database (`SBDB`)**: Catálogo completo de **43.072 Objetos Próximos à Terra (NEOs)** com elementos orbitais, incerteza efeméride (`OCC`), arco observacional e taxonomia espectral real (`Bus-DeMeo` / `Tholen`).
2. **NASA / Caltech JPL — `NHATS` (*Near-Earth Object Human/Robotic Accessible Targets*)**: **7.169 asteroides acessíveis** com cálculos balísticos reais de $\\Delta v$ mínimo ($\\text{km/s}$), duração de missão e janelas viáveis.
3. **USSPACECOM / CelesTrak — Catálogo Oficial `NORAD SATCAT`**: **71.262 objetos espaciais rastreados** desde o Sputnik 1 (`1957–2026`), permitindo classificar fisicamente os corredores orbitais (`LEO`, `MEO`, `GEO`, `HEO/Cislunar`, `Espaço Profundo`) e medir a fração real de satélites ativos vs. detritos espaciais.
4. **Harvard-Smithsonian Center for Astrophysics — `GCAT Launch Log` (J. McDowell)**: Registro histórico completo de **7.418 lançamentos orbitais e interplanetários distintos** distribuídos em **441 modelos de foguetes**, sustentando a modelagem atuarial de confiabilidade (Limite Inferior de Wilson 95% e Bayes Empírico).

**Identidade Visual Oficial do Projeto:** `#160D18` • `#23145B` • `#09456C` • `#026F6E` • `#1CA39E`"""
        ),
        new_code_cell(
            """from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = Path.cwd().resolve()
if not (ROOT_DIR / "src" / "newspace").exists() and (ROOT_DIR.parent / "src" / "newspace").exists():
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
import plotly.io as pio
from IPython.display import HTML, Markdown, display

from newspace.config import DEFAULT_CRITERIA, PALETTE
from newspace.ingestion import load_all_real_datasets
from newspace.database import persist_relational_and_parquet, save_actuarial_scenarios
from newspace.benchmarking import run_complete_benchmark_suite
from newspace.selection_trends import (
    analyze_information_production_trends,
    apply_selection_criteria_and_topsis,
)
from newspace.dashboard import (
    create_notebook_interactive_cockpit,
    export_standalone_custom_dashboard,
    fig_actuarial_scenarios_matrix,
    fig_engine_benchmark,
    fig_information_trends,
    fig_ml_benchmark,
    fig_observability_and_corridors,
    fig_topsis_prospecting_map,
)
from newspace.reporting import generate_technical_decision_report

pio.renderers.default = "plotly_mimetype+notebook_connected"
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)

display(HTML(f'''
<div style="display:flex;gap:10px;align-items:center;background:{PALETTE.cosmic_bg};padding:12px 18px;border-radius:8px;border:1px solid {PALETTE.vibrant_cyan};color:#fff;font-family:Segoe UI,sans-serif;">
  <b>Ambiente Inicializado com Sucesso</b> | Raiz: <code>{ROOT_DIR.name}</code> | Paleta Ativa:
  <span style="background:#160D18;padding:3px 10px;border:1px solid #fff;border-radius:4px;">#160D18</span>
  <span style="background:#23145B;padding:3px 10px;border-radius:4px;">#23145B</span>
  <span style="background:#09456C;padding:3px 10px;border-radius:4px;">#09456C</span>
  <span style="background:#026F6E;padding:3px 10px;border-radius:4px;">#026F6E</span>
  <span style="background:#1CA39E;color:#000;font-weight:700;padding:3px 10px;border-radius:4px;">#1CA39E</span>
</div>
'''))"""
        ),
        new_markdown_cell(
            """---
## 1. Auditoria de Proveniência (SHA-256), Ingestão Empírica e Persistência Relacional

Nesta etapa validamos criptograficamente os arquivos brutos oficiais em `data/raw/manifest_provenance.json`, realizamos a normalização física das entidades e materializamos a persistência dupla:
- **Banco Relacional Transacional (`SQLite` em modo `WAL` + Índices Compostos Cobertos + `ANALYZE` + Integridade Referencial Estrita)** em `data/processed/newspace_analytics.db`.
- **Camada Analítica Colunar (`Apache Parquet`)** em `data/processed/*.parquet` para avaliação OLAP."""
        ),
        new_code_cell(
            """datasets = load_all_real_datasets()
db_counts = persist_relational_and_parquet(datasets)

prov_df = pd.DataFrame(datasets.provenance["datasets"])[
    ["id", "description", "size_mb", "sha256", "verified_at_utc"]
].copy()
prov_df["sha256"] = prov_df["sha256"].str.slice(0, 16) + "..."

display(Markdown("### 1.1 Manifesto de Proveniência Científica Auditada"))
display(prov_df)

summary_entities = pd.DataFrame(
    [{"Tabela Relacional (SQLite & Parquet)": k, "Registros Empíricos Reais": f"{v:,}"} for k, v in db_counts.items()]
)
display(Markdown("### 1.2 Volumetria Persistida com Integridade Referencial Verificada (`PRAGMA foreign_key_check`)"))
display(summary_entities)"""
        ),
        new_markdown_cell(
            """---
## 2. Benchmarking Multidimensional da Mineração e Persistência de Dados

Executamos uma bateria quantitativa rigorosa em **três dimensões complementares**:
1. **Motores de Persistência e Consulta (`OLTP` vs. `OLAP`)**: Comparação de latência mediana, P95, pico de memória (`tracemalloc`), *throughput* e taxa de compressão entre `SQLite (Sem Índices)`, `SQLite (Indexado + WAL)`, `Pandas (In-Memory)`, `Polars (Lazy Parquet)` e `DuckDB (OLAP Parquet)` sob três cargas reais (`W1`: Join de Tráfego Orbital 71k linhas; `W2`: Filtragem Multicritério NASA NHATS 43k linhas; `W3`: Série Histórica Atuarial 7,4k lançamentos).
2. **Mineração Não-Supervisionada (Segmentação Astrodinâmica)**: Avaliação de 5 algoritmos de clusterização sobre os **7.148 asteroides acessíveis NHATS** com métricas `Silhouette`, `Calinski-Harabasz`, `Davies-Bouldin` e custo computacional.
3. **Mineração Supervisionada Preditiva (Risco Atuarial de Lançamento)**: Avaliação em validação cruzada estratificada (`5-Fold CV`) de 5 classificadores sobre **7.418 lançamentos orbitais**, utilizando engenharia de atributos **estritamente cronológica/causal (zero vazamento de futuro)** e priorizando `ROC-AUC` e `Brier Score` (calibração probabilística de sinistro)."""
        ),
        new_code_cell(
            """bench = run_complete_benchmark_suite(datasets)

display(Markdown("### 2.1 Síntese Global dos Motores de Persistência e Consulta Analítica"))
display(bench.engine_summary)

display(Markdown("### 2.2 Detalhamento por Carga de Trabalho (Workloads W1, W2 e W3 — Latência Mediana ms)"))
display(
    bench.engine_workloads.pivot(
        index="engine", columns="workload", values="median_ms"
    ).round(3)
)

display(Markdown("### 2.3 Eficiência de Armazenamento e Compressão em Disco"))
display(bench.storage_footprint)

fig_engine_benchmark(bench).show()"""
        ),
        new_code_cell(
            """display(Markdown("### 2.4 Benchmarking de Clusterização Não-Supervisionada (Asteroides NASA NHATS)"))
display(bench.clustering_benchmark)

cluster_profiles = (
    bench.clustered_nhats.groupby("arquetipo_astrodinamico", as_index=False)
    .agg(
        qtd_asteroides=("spkid", "count"),
        delta_v_mediano_kms=("min_delta_v_kms", "median"),
        duracao_mediana_dias=("duracao_min_dv_dias", "median"),
        excentricidade_mediana=("excentricidade_e", "median"),
        observacoes_medianas=("num_observacoes", "median"),
    )
    .round(3)
)
display(Markdown("#### Perfil Físico dos Arquétipos Minerados Automaticamente"))
display(cluster_profiles)

display(Markdown("### 2.5 Benchmarking de Classificadores Supervisionados Atuariais (5-Fold CV)"))
display(bench.classification_benchmark)

fig_ml_benchmark(bench).show()"""
        ),
        new_markdown_cell(
            """---
## 3. Tendência de Produção de Informações Espaciais (1957–2030) e Gargalos de Negócio

A análise temporal conjunta das três fontes empíricas demonstra:
1. **Mudança Estrutural Exponencial pós-2015 (*Era NewSpace*)**: Crescimento abrupto na produção anual de telemetria orbital (`PAY`), capacidade de lançamento compartilhado (*rideshare* / cargas por foguete) e descoberta astrométrica de NEOs.
2. **Lei Empírica Informação $\\times$ Risco Navegacional**: Correlação estatística de Spearman comprovando que o acúmulo de observações astrométricas colapsa a incerteza orbital (`OCC 9 -> OCC 0`), transformando corpos recém-descobertos em alvos seguráveis.
3. **Bifurcação de Detritos nos Corredores Orbitais & Gargalo Espectroscópico**: Identificação da concentração crítica de objetos inertes (`DEB + R/B`) nos corredores `LEO_SSO_ALTA` e `HEO_CISLUNAR`, bem como da escassez de espectroscopia mineralógica nos NEOs catalogados."""
        ),
        new_code_cell(
            """trends = analyze_information_production_trends(datasets)

display(Markdown("### 3.1 Comparativo Estrutural entre Eras Espaciais (Anos Completos 1957–2025)"))
display(trends.era_comparison)

display(Markdown("### 3.2 Projeção Log-Linear da Produção de Informações (2026–2030)"))
display(trends.forecast_2030)

rho_occ, pval_occ = trends.spearman_obs_vs_occ
gap = trends.characterization_gap
display(Markdown(
    f"**Correlação Empírica Observações vs. Incerteza Orbital (OCC):** Spearman $\\\\rho = {rho_occ:.4f}$ ($p = {pval_occ:.2e}$)  \\n"
    f"**Diagnóstico de Caracterização Espectral NASA JPL:** Apenas **{gap['pct_neos_com_espectro']:.2f}%** "
    f"({gap['neos_com_espectro_real']:,} de {gap['total_neos_catalogados']:,} NEOs) possuem classe espectral medida, "
    f"evidenciando a demanda imediata por missões precursoras de sensoriamento mineral."
))

fig_information_trends(trends).show()"""
        ),
        new_code_cell(
            """display(Markdown("### 3.3 Diagnóstico Físico dos Corredores Orbitais (NORAD SATCAT Atual)"))
display(
    datasets.corridors[
        [
            "id_corredor",
            "regime_orbital",
            "objetos_em_orbita",
            "satelites_ativos",
            "detritos_e_estagios_orbita",
            "razao_detritos_pct",
            "expansao_pos_2021_pct",
            "indice_congestao_orbital",
        ]
    ]
)

fig_observability_and_corridors(datasets, trends).show()"""
        ),
        new_markdown_cell(
            """---
## 4. Critérios de Seleção de Dados do Negócio (MCDA-TOPSIS) e Matriz Atuarial

Aplicamos os **Critérios de Seleção de Dados** alinhados ao modelo de negócio integrado da equipe:
- **Acessibilidade Energética e Operacional:** Solução balística viável no catálogo `NASA NHATS`, $\\Delta v \\le 6,5\\text{ km/s}$ e duração $\\le 450\\text{ dias}$.
- **Qualidade e Maturidade da Informação Orbital:** Código de Incerteza `OCC <= 3` e mínimo de `50` observações astrométricas confirmadas.
- **Porte Físico Mínimo:** Diâmetro físico de referência $\\ge 25\\text{ m}$.
- **Subscrição Atuarial de Lançadores:** Mínimo de `10` voos orbitais, operação ativa na era NewSpace e Limite Inferior de Confiabilidade de Wilson (95%) $\\ge 85\\%$.
- **Ordenação Multicritério (`TOPSIS`)**: Combina eficiência energética ($\\Delta v$ e duração), índice composto de maturidade informacional, escala física e redundância de trajetórias viáveis."""
        ),
        new_code_cell(
            """selection = apply_selection_criteria_and_topsis(datasets)
n_saved_scen = save_actuarial_scenarios(selection.actuarial_scenarios)

display(Markdown("### 4.1 Funil de Seletividade dos Critérios de Negócio"))
display(selection.funnel_stages)

display(Markdown("### 4.2 Top 15 Asteroides Próximos à Terra Recomendados (Ranking MCDA-TOPSIS)"))
display(
    selection.ranked_asteroids[
        [
            "ranking_topsis",
            "spkid",
            "nome_completo",
            "classe_espectral",
            "grupo_mineralogico",
            "min_delta_v_kms",
            "duracao_min_dv_dias",
            "diametro_fisico_ref_m",
            "num_observacoes",
            "codigo_incerteza_occ",
            "indice_maturidade_info",
            "score_topsis",
        ]
    ].head(15)
)

fig_topsis_prospecting_map(selection).show()"""
        ),
        new_code_cell(
            """display(Markdown("### 4.3 Veículos Lançadores Qualificados pelo Critério Atuarial (Wilson 95%)"))
display(
    selection.qualified_launchers[
        [
            "id_veiculo",
            "modelo_veiculo",
            "agencia_principal",
            "pais_estado",
            "total_lancamentos",
            "taxa_sucesso_pct",
            "wilson_lower_95_pct",
            "confiabilidade_bayes_pct",
            "ultimo_voo_ano",
        ]
    ].head(10)
)

display(Markdown(f"### 4.4 Cenários Integrados de Subscrição Atuarial ({n_saved_scen} cenários persistidos no SQLite)"))
display(
    selection.actuarial_scenarios[
        [
            "id_cenario",
            "nome_asteroide",
            "modelo_veiculo",
            "id_corredor_transito",
            "min_delta_v_kms",
            "indice_maturidade_info",
            "confiabilidade_wilson_pct",
            "score_topsis",
            "score_risco_combinado",
            "rating_atuarial",
        ]
    ].head(12)
)

fig_actuarial_scenarios_matrix(selection).show()"""
        ),
        new_markdown_cell(
            """---
## 5. Dashboard Customizável (Simulador Interativo Jupyter + Cockpit Web HTML5 Standalone)

Abaixo você dispõe de **duas interfaces interativas sincronizadas**:
1. **Cockpit Interativo `ipywidgets` no próprio Notebook**: Altere dinamicamente os limites de $\\Delta v$, duração, incerteza `OCC`, observações mínimas, porte físico e os **4 pesos decisórios do TOPSIS** para recalcular em tempo real o portfólio qualificado.
2. **Aplicação Web Standalone Customizável (`reports/newspace_dashboard.html`)**: Exportada automaticamente com motor JavaScript TOPSIS no cliente sobre todos os 7.169 alvos NHATS — pode ser aberta diretamente em qualquer navegador sem servidor ativo e permite exportar CSVs customizados."""
        ),
        new_code_cell(
            """dashboard_html_path = export_standalone_custom_dashboard(datasets, bench, trends, selection)
display(Markdown(f"✅ **Cockpit Web Customizável exportado com sucesso para:** `{dashboard_html_path.relative_to(ROOT_DIR)}`"))

cockpit_widget = create_notebook_interactive_cockpit(datasets)
display(cockpit_widget)"""
        ),
        new_markdown_cell(
            """---
## 6. Geração Automatizada do Relatório Técnico Executivo para Tomada de Decisão

Consolidamos automaticamente todas as métricas empíricas obtidas na execução em um **Relatório Técnico Executivo** parametrizado (salvo em `reports/relatorio_tecnico_decisao.md` e em versão formatada para apresentação/impressão PDF em `reports/relatorio_tecnico_decisao.html`), exibido integralmente abaixo:"""
        ),
        new_code_cell(
            """md_report_path, html_report_path = generate_technical_decision_report(
    datasets, bench, trends, selection
)
display(Markdown(
    f"📄 **Relatórios gerados:** `{md_report_path.relative_to(ROOT_DIR)}` | `{html_report_path.relative_to(ROOT_DIR)}`"
))
display(Markdown("---"))
display(Markdown(md_report_path.read_text(encoding="utf-8")))"""
        ),
    ]

    nb.cells = cells
    return nb


def execute_notebook_in_process(nb: nbformat.NotebookNode, workdir: Path) -> None:
    """Execute all code cells in-process via IPython InteractiveShell, capturing rich MIME outputs."""
    prev_cwd = Path.cwd()
    os.chdir(workdir)
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    pub = NotebookCapturePublisher(shell=shell)
    shell.display_pub = pub

    exec_idx = 1
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            pub.outputs = []
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                res = shell.run_cell(cell.source, store_history=True)

            if res.error_before_exec is not None:
                raise RuntimeError(f"Syntax/Setup error in cell #{exec_idx}: {res.error_before_exec}")
            if res.error_in_exec is not None:
                raise RuntimeError(f"Runtime error in cell #{exec_idx}: {res.error_in_exec}")

            cell_outputs = []
            out_txt = stdout_buf.getvalue()
            if out_txt:
                cell_outputs.append(new_output(output_type="stream", name="stdout", text=out_txt))

            cell_outputs.extend(pub.outputs)

            if res.result is not None:
                fmt_dict, md_dict = shell.display_formatter.format(res.result)
                cell_outputs.append(
                    new_output(
                        output_type="execute_result",
                        execution_count=exec_idx,
                        data=fmt_dict,
                        metadata=md_dict,
                    )
                )

            cell.execution_count = exec_idx
            cell.outputs = cell_outputs
            print(f"  [OK] Cell #{exec_idx} executed ({len(cell_outputs)} rich outputs captured)", flush=True)
            exec_idx += 1
    finally:
        InteractiveShell.clear_instance()
        os.chdir(prev_cwd)


def build_and_run_notebook() -> Path:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    nb_path = NOTEBOOKS_DIR / "newspace_benchmark_analytics.ipynb"

    try:
        nb = create_notebook_structure()
        print("Executing all notebook cells in-process via IPython...", flush=True)
        execute_notebook_in_process(nb, NOTEBOOKS_DIR)
        nbformat.validate(nb)
        nbformat.write(nb, nb_path)
        size_kb = round(nb_path.stat().st_size / 1024.0, 1)
        print(f"Successfully validated and saved executed notebook ({size_kb} KB): {nb_path}", flush=True)
    finally:
        shutil.rmtree(PROJECT_ROOT / ".jupyter_runtime", ignore_errors=True)
    return nb_path


if __name__ == "__main__":
    build_and_run_notebook()
