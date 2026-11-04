# 🛰️ NewSpace Analytics — Benchmarking de Mineração de Dados, Tendências de Produção de Informações e Suporte à Decisão

Aplicação completa em **Notebook Python** apoiada por pacotes modulares (`src/newspace/`) para realizar o **benchmarking multidimensional da mineração e persistência de dados**, modelar a **tendência histórica e projetada de produção de informações espaciais (1957–2030)**, aplicar **critérios de seleção de dados multicritério (MCDA-TOPSIS) e risco atuarial** em um **dashboard customizável**, e gerar automaticamente o **Relatório Técnico para Tomada de Decisão**.

---

## 🛡️ Governança e Proveniência Científica (Zero Dados Fictícios)

Nenhum registro sintético, aleatório ou hipotético foi criado. Todos os **122.200 registros normalizados** provêm diretamente de quatro repositórios científicos públicos oficiais, verificados criptograficamente via manifesto SHA-256 (`data/raw/manifest_provenance.json`):

| Fonte Oficial | Instituição | Volume Real | Uso no Negócio NewSpace |
| :--- | :--- | ---: | :--- |
| **Small-Body Database (`SBDB`)** | NASA / Caltech JPL | **43.072 NEOs** | Elementos orbitais, incerteza (`OCC`), observações astrométricas (`1893–2026`) e espectroscopia |
| **NHATS Accessible NEAs** | NASA / Caltech JPL | **7.169 alvos** | Soluções balísticas reais de ida e volta ($\Delta v$ em $\text{km/s}$, duração e janelas viáveis) |
| **Catálogo Oficial `SATCAT`** | USSPACECOM / CelesTrak | **71.262 objetos** | Tráfego real em corredores orbitais (`LEO`, `MEO`, `GEO`, `HEO/Cislunar`), satélites ativos e detritos |
| **GCAT Orbital Launch Log** | Harvard-Smithsonian CfA | **7.418 lançamentos** | Histórico completo de 441 modelos de foguetes (`1957–2026`) para confiabilidade atuarial (Wilson 95%) |

---

## 🗂️ Estrutura Arquitetural do Projeto

```text
New space trabalho final/
├── notebooks/
│   └── newspace_benchmark_analytics.ipynb   # Notebook Python Principal (pré-executado + cockpit interativo)
├── reports/
│   ├── newspace_dashboard.html              # Dashboard Web Customizável Standalone (TOPSIS em tempo real no navegador)
│   ├── relatorio_tecnico_decisao.md         # Relatório Técnico Executivo para Tomada de Decisão (Markdown)
│   └── relatorio_tecnico_decisao.html       # Relatório Técnico Formatado (identidade visual oficial / pronto p/ PDF)
├── src/newspace/
│   ├── config.py                            # Paleta oficial (#160D18, #23145B, #09456C, #026F6E, #1CA39E) e constantes
│   ├── ingestion.py                         # Parsing físico, validação SHA-256 e cálculo de Wilson/Bayes
│   ├── database.py                          # Persistência Relacional SQLite (WAL + Índices Cobertos + ANALYZE) e Parquet
│   ├── benchmarking.py                      # Benchmarking: (1) 5 Motores OLTP/OLAP, (2) Clusterização, (3) ML Atuarial
│   ├── selection_trends.py                  # Séries temporais 1957–2030, funil de critérios, TOPSIS e Rating Atuarial
│   ├── dashboard.py                         # Gráficos Plotly, widgets Jupyter (ipywidgets) e gerador HTML5 client-side
│   └── reporting.py                         # Síntese executiva automatizada orientada a dados
├── data/
│   ├── raw/                                 # Bases brutas oficiais + manifest_provenance.json (SHA-256)
│   └── processed/                           # Banco newspace_analytics.db (SQLite) + arquivos colunares .parquet
├── scripts/
│   ├── fetch_real_data.py                   # Download auditado das APIs públicas da NASA JPL, CelesTrak e GCAT
│   └── build_and_execute_notebook.py        # Construção e execução automatizada do Notebook via nbclient
├── tests/
│   └── test_pipeline.py                     # Suíte automatizada de testes (pytest)
├── pyproject.toml                           # Gerenciamento moderno via uv / PEP 621
└── requirements.txt                         # Compatibilidade direta com pip / Jupyter / Colab
```

---

## 🚀 Como Executar e Apresentar

### 1. Visualização Imediata (Sem precisar rodar servidor)
- **Notebook Completo:** Abra `notebooks/newspace_benchmark_analytics.ipynb` no **VS Code** ou **JupyterLab** (todas as células já estão executadas com tabelas, gráficos interativos e relatório embutidos).
- **Dashboard Web Customizável:** Abra `reports/newspace_dashboard.html` diretamente no navegador (Chrome/Edge). Os controles deslizantes de $\Delta v$, duração, incerteza `OCC`, observações mínimas, filtro mineralógico e os **4 pesos do TOPSIS** recalculam instantaneamente os 7.169 alvos da NASA no navegador e permitem exportar CSV!
- **Relatório Técnico:** Abra `reports/relatorio_tecnico_decisao.html` (pressione `Ctrl+P` no navegador caso deseje salvar em PDF para entrega acadêmica).

### 2. Reexecução Completa do Ambiente Python
Com `uv` (já configurado no ambiente):
```powershell
# Sincronizar ambiente virtual
uv sync

# Executar suíte de testes de integridade e qualidade
.venv\Scripts\pytest.exe -v

# Reexecutar o notebook de ponta a ponta e regerar dashboard + relatórios
.venv\Scripts\python.exe -X utf8 scripts\build_and_execute_notebook.py

# Abrir JupyterLab interativo
.venv\Scripts\jupyter.exe lab notebooks\newspace_benchmark_analytics.ipynb
```
*(Alternativamente, com `pip` tradicional: `pip install -r requirements.txt`)*
