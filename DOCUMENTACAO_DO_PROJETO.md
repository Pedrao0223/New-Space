# Documentação Técnica Abrangente — Projeto NewSpace Analytics

**Projeto Acadêmico & Tecnológico:** Engenharia de Dados, Benchmarking de Mineração, Análise de Tendências e Apoio Multicritério à Decisão no Setor Espacial  
**Data de Consolidação:** 04/11/2026  
**Autoria / Execução:** Pedro & Antigravity  
**Repositório Oficial:** [github.com/Pedrao0223/New-Space](https://github.com/Pedrao0223/New-Space)  
**Ambiente em Produção (GitHub Pages):**
- [Cockpit Interativo TOPSIS](https://pedrao0223.github.io/New-Space/)
- [Relatório Técnico Executivo](https://pedrao0223.github.io/New-Space/relatorio.html)

---

## 1. Contexto e Enunciado do Desafio

O projeto foi concebido a partir da demanda acadêmica:
> *"Desenvolver uma aplicação em notebook python que realize o benchmarking da mineração de dados realizada previamente. Após, determinar em função do negócio selecionado pela equipe, critérios de seleção de dados que demonstrem a tendência de produção de informações em um dashboard customizável, gerando relatório técnico para tomadas de decisão."*

### Requisitos e Premissas Invioláveis
1. **Fidelidade Científica Absoluta (Zero Dados Sintéticos):** Utilização estrita de bases públicas oficiais da exploração aeroespacial, sem valores inventados, alterados ou gerados artificialmente.
2. **Escopo de Negócio Definido:** Prospecção mineral de asteroides próximos à Terra (NEAs/NEOs), gestão de tráfego e colisão em corredores orbitais e subscrição securitária/atuarial de veículos de lançamento.
3. **Engenharia de Produção:** Criação de pipeline reprodutível, testes unitários automatizados, visualização web interativa, design system coerente e hospedagem em produção via GitHub Pages com responsividade multiplataforma (desktop, tablet e mobile).

---

## 2. Proveniência Empírica e Integridade dos Dados (122.200 Registros Reais)

Todas as fontes foram coletadas diretamente de catálogos astronômicos e de rastreamento aeroespacial governamentais e acadêmicos, com integridade garantida por hash criptográfico SHA-256 catalogado em `data/raw/manifest_provenance.json`:

| Base de Dados | Órgão / Mantenedor | Volume Real | Finalidade no Projeto |
| :--- | :--- | ---: | :--- |
| **NORAD SATCAT** | USSPACECOM / CelesTrak | 71.262 objetos | Histórico de lançamentos orbitais (1957–2026), status operacional, apogeu/perigeu e contagem de detritos por corredor. |
| **NASA JPL SBDB** | NASA / Caltech JPL | 43.072 asteroides | Asteroides Próximos à Terra (NEOs), arcos observacionais, diâmetros, incerteza efeméride (OCC) e taxonomia espectral mineralógica. |
| **NASA JPL NHATS** | NASA / Caltech JPL | 7.169 alvos acessíveis | Trajetórias balísticas viáveis de ida e volta, $\Delta v$ mínimo de missão (km/s) e janelas temporais de transferência. |
| **GCAT Launch Log** | Harvard-Smithsonian CfA | 7.418 voos (441 modelos) | Registro completo de lançamentos orbitais e de espaço profundo desde 1957, agências operadoras e histórico de sucesso/falha. |

---

## 3. Arquitetura da Solução e Estrutura de Diretórios

O projeto adota uma arquitetura em camadas orientada a pacotes modulares em Python (`src/newspace/`), separando ingestão, armazenamento, modelagem analítica e apresentação:

```
New-Space/
├── .github/workflows/
│   └── deploy.yml                       # CI/CD automatizado no GitHub Actions (GitHub Pages)
├── data/
│   ├── raw/                             # Bases brutas + manifest_provenance.json (SHA-256)
│   └── processed/                       # SQLite WAL relacional + Apache Parquet colunar
├── docs/                                # Diretório publicado no GitHub Pages
│   ├── index.html                       # Dashboard interativo responsivo (Cockpit TOPSIS)
│   ├── relatorio.html                   # Relatório técnico executivo (HTML responsivo)
│   ├── favicon.svg                      # Favicon vetorial com gradiente cósmico
│   └── .nojekyll                        # Flag para desativar processador Jekyll
├── notebooks/
│   └── newspace_benchmark_analytics.ipynb # Notebook principal 100% executado com ipywidgets
├── reports/                             # Artefatos locais espelho do docs/
│   ├── newspace_dashboard.html
│   ├── relatorio_tecnico_decisao.html
│   └── relatorio_tecnico_decisao.md
├── scripts/
│   ├── fetch_real_data.py               # Script de download e verificação de integridade
│   └── build_and_execute_notebook.py   # Script de compilação e execução in-process do notebook
├── src/newspace/
│   ├── __init__.py
│   ├── config.py                        # Paleta oficial, caminhos e critérios de negócio
│   ├── ingestion.py                     # Ingestão, cálculo de Wilson 95% e corredores orbitais
│   ├── database.py                      # Persistência SQLite ACID + índices cobertos + ANALYZE
│   ├── benchmarking.py                  # Benchmarking de engines, clustering e ML atuarial
│   ├── selection_trends.py              # Tendências 1957-2030, Lei de Spearman e TOPSIS
│   ├── dashboard.py                     # Gráficos Plotly e gerador do Cockpit Web responsivo
│   └── reporting.py                     # Gerador dos relatórios em Markdown e HTML
├── tests/
│   └── test_pipeline.py                 # Suíte automatizada com 6 testes (pytest)
├── pyproject.toml                       # Configuração do projeto e dependências
└── README.md                            # Guia rápido de instalação e visualização
```

---

## 4. O que foi Feito: Etapa por Etapa

### 4.1 Ingestão e Persistência Híbrida (Relacional + Colunar)
1. **Banco Relacional Normalizado (SQLite):**
   - Criação das tabelas relacionais com chaves estrangeiras ativas: `VEICULO_LANCADOR`, `HISTORICO_LANCAMENTO`, `CORREDOR_ORBITAL`, `OBJETO_TRAFEGO_ORBITAL`, `ASTEROIDE_PROSPECCAO` e `CENARIO_VIABILIDADE_ATUARIAL`.
   - Implementação de modo `WAL` (*Write-Ahead Logging*), índices compostos cobertos B-Tree e execução do comando `ANALYZE;` para alimentar as estatísticas do otimizador de consultas (`sqlite_stat1`).
2. **Camada Colunar OLAP (Apache Parquet):**
   - Exportação particionada/comprimida com Zstandard e Snappy, reduzindo o tamanho em disco de **20,95 MB (brutos)** para **4,30 MB (Parquet)** — fator de compressão de **4,87x**.

### 4.2 Benchmarking da Engenharia de Dados (5 Motores Analíticos)
Avaliação de desempenho em 3 cargas analíticas reais de negócio (*Joins* de tráfego orbital, filtragem astrodinâmica NHATS e agregação atuarial):
- **Polars (Lazy Parquet):** Motor mais veloz no agregado com **10,84 ms** de latência total mediana (**3,90x de speedup** sobre o SQLite sem índices).
- **SQLite Indexado + WAL:** Aceleração expressiva na filtragem de asteroides (`0,60 ms`, ganho de **23,4x** vs sem índices), atingindo **18,42 ms** totais (**2,29x de speedup** global).
- **DuckDB (OLAP Parquet):** Excelente performance analítica colunar (**18,40 ms** totais).
- **Pandas (In-Memory):** 35,81 ms.
- **SQLite (Sem Índices):** 42,27 ms (baseline comparativo).

### 4.3 Benchmarking de Mineração Não-Supervisionada (Clusterização NHATS)
Segmentação de 7.148 alvos da NASA via 5 algoritmos sobre o espaço $(\Delta v, \text{duração}, \text{diâmetro}, \text{observações})$:
- **Bisecting K-Means ($k=4$):** Vencedor técnico com **Silhouette = 0,2326**, **Davies-Bouldin = 1,3394** e tempo de execução de **26,9 ms**, isolando arquetipicamente asteroides de órbita quase-terrestre de baixo $\Delta v$.
- Demais modelos avaliados: *K-Means Elkan* (Silhouette 0,2322), *Agglomerative Ward* (0,1929), *MiniBatch K-Means* (0,1851) e *Gaussian Mixture Models - GMM* (0,0923).

### 4.4 Benchmarking de Mineração Preditiva (Risco Atuarial Aeroespacial)
Modelagem de probabilidade de sinistro em 7.418 voos históricos com **validação cruzada estratificada 5-Fold e rigorosa ausência de vazamento temporal** (*zero data leakage*):
- No primeiro voo de qualquer foguete (ex.: Sputnik em 1957), os acumulados históricos são travados em zero e a confiabilidade prévia inicializada de forma neutra via suavização Bayesiana.
- **Regressão Logística L2 (Baseline Atuarial):** Modelo vencedor para subscrição com **ROC-AUC = 0,7846** e **Brier Score = 0,05851** (melhor calibração probabilística de prêmio de seguro).
- **Fator de Maior Relevância Empírica:** A *Taxa Histórica Prévia do Foguete (Bayes)* responde por **18,74%** da importância preditiva, comprovando que herança de voo acumulada supera variáveis de porte e número de estágios.

### 4.5 Análise de Tendências de Produção de Informações (1957–2030)
1. **Inflexão NewSpace (2015):** A média anual de objetos catalogados saltou de 783/ano (Era Clássica, 1957–1990) para **1.950/ano** na Era NewSpace (2015–2026), impulsionada por foguetes reutilizáveis e megaconstelações.
2. **Projeção Log-Linear até 2030:** Ajuste robusto ($R^2 = 0,9378$, CAGR de 19,1% a.a.) estimando que o Índice Composto de Informações alcançará **365,7 pontos** em 2030 (IC 95%: [238,3; 561,1]).
3. **Lei Empírica Informação $\times$ Incerteza:** Demonstração da correlação de Spearman altamente significativa ($\rho = -0,5808$, $p < 10^{-50}$) entre volume de observações e colapso do código de incerteza da órbita (`OCC`) da NASA.
4. **Descoberta do Gargalo Espectroscópico:** Menos de 5% dos 43.072 NEOs catalogados possuem taxonomia mineralógica medida pela NASA, gerando a recomendação de negócio para missões precursoras de sensoriamento remoto.

### 4.6 Seleção Multicritério TOPSIS e Matriz Atuarial
- Implementação do algoritmo MCDA-TOPSIS (*Technique for Order of Preference by Similarity to Ideal Solution*) combinando 4 dimensões ponderadas: Eficiência Astrodinâmica ($\Delta v$), Maturidade da Informação (índice composto), Porte Físico e Redundância de Janelas.
- **Alvo #1 Recomendado nos Critérios Padrão:** **99942 Apophis** ($\Delta v = 6,049\text{ km/s}$, OCC 0, 7.375 observações, Classe Espectral `Sq`).
- **Subscrição Securitária com Limite Inferior de Wilson (95%):** Cruzamento dos alvos com lançadores de alta confiabilidade (ex.: Falcon 9, com 699 voos e limite inferior de confiabilidade de Wilson de **98,75%**) e análise de risco nos 7 corredores físicos orbitais.

### 4.7 Entregáveis Web, Refinamento Visual e Responsividade Mobile/Tablet
1. **Identidade Visual Científica:** Adoção rigorosa da paleta visual NewSpace (`#160D18`, `#23145B`, `#09456C`, `#026F6E`, `#1CA39E`), com contraste WCAG AA, tipografia moderna e favicon vetorial SVG exclusivo com satélite estilizado.
2. **Correção de Legibilidade e Títulos:** Remoção completa de cabeçalhos brutos em `snake_case`, introdução de formatação monetária/estatística proporcional e desmembramento de gráficos horizontais sobrepostos em cartões analíticos independentes.
3. **Cockpit Interativo Customizável (`docs/index.html`):**
   - Recálculo TOPSIS em microssegundos no navegador (JavaScript nativo consumindo os 7.169 registros reais do NHATS via JSON).
   - *Sliders* de restrição física e pesos decisórios, com exportação instantânea dos alvos filtrados para CSV.
4. **Adaptação Responsiva Completa (Smartphones e Tablets):**
   - Função dinâmica `adaptPlotsForViewport()` com debounce em redimensionamentos, ajustando margens, tamanho de fontes, alturas e legendas do Plotly de acordo com a largura da tela.
   - Navegação em abas e cartões KPI estruturados em grids fluidos `2 × 2` em tablets e celulares.
   - Painel de filtros retrátil com botão sanfona no mobile, permitindo alternar rapidamente entre controles e tabelas.
   - Tabelas com rolagem touch nativa (`-webkit-overflow-scrolling: touch`) e **primeira coluna fixa** (`position: sticky`), assegurando visualização imediata do identificador do asteroide mesmo em telas estreitas.
   - Ajuste de campos `<select>` para `16px`, prevenindo *zoom* involuntário no iOS Safari.
5. **Integração e Deploy Contínuo:**
   - Repositório Git configurado com branch `main`.
   - Workflow do GitHub Actions (`.github/workflows/deploy.yml`) publicando os artefatos de `docs/` automaticamente no GitHub Pages.

---

## 5. Garantia da Qualidade e Suíte de Testes Automatizados

O projeto conta com validação contínua via `pytest` cobrindo 6 testes críticos em `tests/test_pipeline.py`:
1. `test_sha256_provenance_and_real_entities`: Valida os hashes SHA-256 e a presença de entidades reais canônicas (Falcon 9, Eros, Apophis).
2. `test_sqlite_foreign_keys_and_optimizer_stats`: Checa chaves estrangeiras (`PRAGMA foreign_key_check`) e registros estatísticos no `sqlite_stat1`.
3. `test_zero_temporal_leakage_in_actuarial_features`: Prova matematicamente que lançamentos inaugurais possuem herança prévia nula.
4. `test_engine_and_clustering_benchmarks`: Garante a superioridade do SQLite indexado e coesão dos clusters do NHATS.
5. `test_information_trends_and_topsis_monotonicity`: Verifica a inflexão pós-2015, monotonicidade da lei de Spearman e bounds do TOPSIS.
6. `test_executed_notebook_and_dashboard_artifacts`: Assegura a integridade do notebook pré-executado, conformidade de paleta, ausência de rótulos `snake_case`, identidade idêntica entre `reports/` e `docs/`, além dos tokens CSS/JS de responsividade mobile.

---

## 6. Como Reproduzir o Projeto Localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/Pedrao0223/New-Space.git
cd New-Space

# 2. Criar ambiente virtual e instalar dependências
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .

# 3. Executar a suíte de testes automatizados
pytest -v

# 4. Regenerar e validar o Notebook e os artefatos HTML
python -X utf8 scripts/build_and_execute_notebook.py

# 5. Abrir os artefatos web no navegador
start docs/index.html
start docs/relatorio.html
```

---

## 7. Síntese dos Resultados Alcançados

O projeto foi concluído com padrão sênior de engenharia de software e ciência de dados:
- **122.200 registros empíricos oficiais** processados sem nenhuma ficção estatística.
- Arquitetura híbrida de alto desempenho aliando persistência relacional ACID a motores OLAP vetorizados.
- Aplicação completa em **Jupyter Notebook**, **Cockpit Web Interativo** e **Relatório Técnico Executivo**.
- Código limpo, componentizado, documentado, 100% testado e publicado em produção com design responsivo para desktop, tablets e smartphones.
