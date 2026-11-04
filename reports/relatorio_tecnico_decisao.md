# Relatório Técnico Executivo para Tomada de Decisão — Operações NewSpace

**Data de Geração:** 04/11/2026 18:23  
**Escopo do Negócio:** Prospecção Mineral de Asteroides Próximos à Terra (NEAs), Gestão de Risco em Corredores Orbitais e Subscrição Atuarial Aeroespacial  
**Política de Integridade Científica:** 100% Dados Públicos Reais Auditados via SHA-256 (122,200 registros empíricos processados — zero registros sintéticos ou fictícios)

---

## 1. Sumário Executivo e Proveniência Empírica

Este relatório consolida os resultados quantitativos do **Benchmarking de Mineração e Persistência de Dados**, da **Modelagem de Tendências de Produção de Informações (1957–2030)** e da **Seleção Multicritério (MCDA-TOPSIS) integrada ao Risco Atuarial** para orientar decisões estratégicas de investimento, engenharia de missão e subscrição de apólices no ecossistema NewSpace.

| Base Oficial Auditada | Instituição Mantenedora | Volume Real Processado | Escopo Técnico |
| :--- | :--- | ---: | :--- |
| **NORAD SATCAT** | USSPACECOM / CelesTrak | 71,262 objetos | Rastreamento orbital completo (1957–2026), status operacional, apogeu/perigeu e detritos |
| **Small-Body DB (SBDB)** | NASA / Caltech JPL | 43,072 asteroides | Elementos orbitais, incerteza efeméride (OCC), arco observacional e taxonomia espectral |
| **NHATS Trajectory DB** | NASA / Caltech JPL | 7,169 alvos viáveis | Soluções balísticas de ida e volta, $\Delta v$ mínimo (km/s) e janelas de lançamento |
| **GCAT Launch Log** | Harvard-Smithsonian CfA | 7,418 lançamentos (441 modelos) | Histórico completo de lançamentos orbitais e de espaço profundo desde o Sputnik 1 |

---

## 2. Diagnóstico do Benchmarking da Mineração e Persistência de Dados

### 2.1 Arquitetura de Persistência e Motores de Consulta Analítica (OLTP vs OLAP)

A evolução do modelo relacional da Fase 1 foi avaliada frente a motores analíticos vetorizados modernos em três cargas representativas de negócio (joins de tráfego orbital, filtragem astrodinâmica NHATS e agregação histórica atuarial):

| engine                  |   latencia_total_mediana_ms |   latencia_media_ms |   p95_medio_ms |   pico_memoria_max_kb |   throughput_medio_krows_s |   speedup_vs_sqlite_sem_indice |
|:------------------------|----------------------------:|--------------------:|---------------:|----------------------:|---------------------------:|-------------------------------:|
| Polars (Lazy Parquet)   |                      10.922 |               3.6   |          3.779 |                   5   |                    11008.5 |                           3.88 |
| SQLite (Indexado + WAL) |                      17.371 |               5.789 |          5.829 |                  20.7 |                    27270   |                           2.44 |
| DuckDB (OLAP Parquet)   |                      18.344 |               6.14  |          6.442 |                 218.8 |                     6239.9 |                           2.31 |
| Pandas (In-Memory)      |                      34.314 |              11.43  |         11.584 |                4260.8 |                     3605.2 |                           1.23 |
| SQLite (Sem Índices)    |                      42.371 |              14.16  |         14.346 |                  20.7 |                     2791.3 |                           1    |

**Conclusões de Engenharia de Dados:**
- **Motor Analítico Vencedor:** **Polars (Lazy Parquet)** atingiu latência total mediana de **10.92 ms** (**3.88x mais rápido** que o SQLite relacional sem índices).
- **Validação da Indexação Relacional:** A adição de índices compostos B-Tree e modo `WAL` no SQLite acelerou as consultas relacionais em **2.44x** (42.37 ms $\rightarrow$ 17.37 ms), validando o SQLite indexado para persistência transacional (ACID + chaves estrangeiras) em conjunto com **Apache Parquet (4.30 MB vs 20.95 MB brutos — compressão de 4.87x)** para analytics em tempo real.

### 2.2 Benchmarking de Algoritmos de Mineração de Dados (Clusterização e Risco Atuarial)

#### A. Segmentação Não-Supervisionada de Asteroides Acessíveis (NASA NHATS)
| algoritmo                        |   amostras_nhats |   silhouette_score |   calinski_harabasz |   davies_bouldin |   tempo_execucao_ms |   pico_memoria_kb |
|:---------------------------------|-----------------:|-------------------:|--------------------:|-----------------:|--------------------:|------------------:|
| Bisecting K-Means (k=4)          |             7148 |             0.2326 |              2046.7 |           1.3394 |               27.91 |            1356.3 |
| K-Means (Elkan, k=4)             |             7148 |             0.2322 |              2141.6 |           1.3189 |              189.39 |            1330.4 |
| Agglomerative Ward (k=4)         |             7148 |             0.1929 |              1630.4 |           1.2465 |             1366.53 |          225158   |
| MiniBatch K-Means (k=4)          |             7148 |             0.1851 |              2005   |           1.5065 |               42.86 |             942.1 |
| Gaussian Mixture (GMM Full, k=4) |             7148 |             0.0923 |              1372.7 |           2.0051 |              239.76 |            1889.3 |

- O algoritmo **Bisecting K-Means (k=4)** obteve a maior coesão estrutural (**Silhouette = 0.2326**, **Davies-Bouldin = 1.3394** em **27.9 ms**), isolando o agrupamento de alvos de órbita quase-terrestre de baixo $\Delta v$ dos objetos de alta excentricidade.

#### B. Modelagem Preditiva Supervisionada de Sinistro de Lançamento (Validação Cruzada Estratificada 5-Fold, Sem Vazamento Temporal)
| modelo_atuarial                            |   lancamentos_avaliados |   taxa_base_sinistro_pct |   roc_auc |   pr_auc |   brier_score_calibracao |   log_loss |   f1_weighted |   tempo_cv_5fold_ms |
|:-------------------------------------------|------------------------:|-------------------------:|----------:|---------:|-------------------------:|-----------:|--------------:|--------------------:|
| Regressão Logística L2 (Baseline Atuarial) |                    7418 |                     7.28 |    0.7846 |   0.3093 |                  0.05851 |     0.2175 |        0.9032 |                54.4 |
| Extra Trees (200 Árvores)                  |                    7418 |                     7.28 |    0.7762 |   0.3354 |                  0.0575  |     0.216  |        0.9036 |              1470.7 |
| Rede Neural MLP (64x32 ReLU)               |                    7418 |                     7.28 |    0.7707 |   0.2853 |                  0.05973 |     0.2231 |        0.9036 |              2234.9 |
| Random Forest (200 Árvores)                |                    7418 |                     7.28 |    0.7678 |   0.3184 |                  0.05809 |     0.2182 |        0.9102 |              2180.4 |
| HistGradientBoosting (GBDT)                |                    7418 |                     7.28 |    0.7576 |   0.3113 |                  0.05876 |     0.2263 |        0.9121 |              1133   |

- **Modelo Atuarial Recomendado:** **Regressão Logística L2 (Baseline Atuarial)** entregou **ROC-AUC = 0.7846** e **Brier Score = 0.05851** (calibração probabilística essencial para precificação de prêmios securitários).
- **Principal Fator Empírico de Risco:** *Taxa Histórica Prévia do Foguete (Bayes)* responde por **18.74%** do poder explicativo do modelo, demonstrando que herança real acumulada e histórico prévio superam variáveis isoladas de porte de carga.

---

## 3. Tendência de Produção de Informações e Gargalos Estratégicos (1957–2030)

A comparação estrutural entre as três eras espaciais comprova uma mudança de regime exponencial na geração de telemetria orbital e astrometria de espaço profundo:

| era_espacial                            |   anos_completos |   media_anual_obj_orbitais |   media_anual_cargas_pay |   media_anual_neos_descobertos |   media_cargas_por_foguete |   taxa_sucesso_media_pct |   indice_info_medio |
|:----------------------------------------|-----------------:|---------------------------:|-------------------------:|-------------------------------:|---------------------------:|-------------------------:|--------------------:|
| 1. Era Estatal Clássica (1957–1990)     |               34 |                     783.18 |                   121.91 |                          12.09 |                       1.71 |                    86.09 |               49.29 |
| 2. Transição Comercial (1991–2014)      |               24 |                     804.42 |                   136.46 |                         544.92 |                       1.98 |                    93.8  |               83.45 |
| 3. Era NewSpace Exponencial (2015–2026) |               11 |                    1949.73 |                  1563.36 |                        2475.73 |                       8.87 |                    94.81 |              340.82 |

### Principais Evidências Quantitativas para Tomada de Decisão:
1. **Explosão de Telemetria na Era NewSpace (2015–2026):** A média anual de objetos orbitais catalogados saltou de **783/ano** na Era Clássica para **1950/ano** na Era NewSpace, enquanto a descoberta anual de asteroides próximos à Terra cresceu de **12.1/ano** para **2476/ano**.
2. **Projeção 2030 (Ajuste Log-Linear, $R^2 = 0.9691$, CAGR 2015–2025 = 16.95% a.a.):** O Índice Composto de Produção de Informações (Base 2010 = 100) projeta atingir **1549.8** até 2030 (IC 95%: [1266.3, 1896.9]), exigindo arquiteturas OLAP colunares (Polars/DuckDB) para ingestão contínua de efemérides e telemetria SSA (*Space Situational Awareness*).
3. **Lei Empírica Informação $\times$ Risco Orbital:** Verificou-se correlação monotônica altamente significativa (**Spearman $\rho = -0.7274$**) entre o volume de observações astrométricas e a redução do código de incerteza orbital da NASA (`OCC`). Mais observações convertem diretamente risco epistêmico em precisão navegacional segurável.
4. **O Grande Gargalo Espectroscópico (Oportunidade de Negócio Imediata):** Embora existam **43,072 NEOs** com órbitas rastreadas (7,169 acessíveis via NHATS), **apenas 0.76% (327 objetos)** possuem espectroscopia mineralógica confirmada na base oficial da NASA JPL, e somente **2.94%** possuem diâmetro físico medido por radar/infravermelho!
   - **Implicação Estratégica:** Antes de comprometer sondas pesadas de lavra mineral, o maior retorno sobre informação está na operação/venda de dados de **nanossondas precursoras de reconhecimento espectral** para alvos NHATS de baixo $\Delta v$.

---

## 4. Análise de Corredores Orbitais e Critérios de Seleção Multicritério (TOPSIS)

### 4.1 Diagnóstico Empírico dos Corredores de Tráfego Orbital
- **Bifurcação de Risco em LEO:** O corredor `LEO_CONSTELACAO` (450–650 km) abriga **13,483 objetos em órbita**, sendo **12,599 satélites ativos manobráveis** (5.2% detritos). Em contraste, `LEO_SSO_ALTA` (650–2.000 km) possui volume similar (**13,511 objetos**), porém composto por **74.5% de detritos e estágios inertes (10,072 objetos não manobráveis)** — índice de congestão **89.8/100**.
- **Alerta Crítico no Corredor de Escape `HEO_CISLUNAR`:** Das **1,503 estruturas em órbita** no corredor elíptico/cislunar obrigatório para missões a asteroides, **84.0% (1,262 objetos)** são estágios superiores descartados (`R/B`) e detritos cruzando o perigeu terrestre. As janelas de injeção translunar/interplanetária devem evitar estritamente estacionamento prolongado acima de 650 km.

### 4.2 Funil de Seleção de Dados Aplicado ao Portfólio
| etapa_funil                                            |   candidatos |   retencao_global_pct |
|:-------------------------------------------------------|-------------:|----------------------:|
| 1. Catálogo Global NASA SBDB (NEOs)                    |        43072 |               100     |
| 2. Solução Balística Viável (NASA NHATS)               |         7169 |                16.644 |
| 3. Janela Energética (Δv ≤ 6.5 km/s & Duração ≤ 450 d) |         1042 |                 2.419 |
| 4. Qualidade Informacional (OCC ≤ 3 & Obs ≥ 50)        |          174 |                 0.404 |
| 5. Porte Físico Operacional (Diâmetro Ref ≥ 25 m)      |           92 |                 0.214 |
| 6. Portfólio Qualificado Final (Pós-Filtro Espectral)  |           92 |                 0.214 |

### 4.3 Top 10 Asteroides Recomendados para Prospecção (Ranking MCDA-TOPSIS)
|   ranking_topsis | pdes       | nome_completo            | classe_espectral    |   min_delta_v_kms |   duracao_min_dv_dias |   diametro_fisico_ref_m |   num_observacoes |   codigo_incerteza_occ |   score_topsis |
|-----------------:|:-----------|:-------------------------|:--------------------|------------------:|----------------------:|------------------------:|------------------:|-----------------------:|---------------:|
|                1 | 99942      | 99942 Apophis (2004 MN4) | Sq                  |             6.049 |                   354 |                340      |              7370 |                      0 |          60.62 |
|                2 | 2009 HC    | (2009 HC)                | NAO_ESPECTROGRAFADO |             4.44  |                   362 |                 42.6615 |               209 |                      0 |          57.25 |
|                3 | 2014 MF18  | (2014 MF18)              | NAO_ESPECTROGRAFADO |             6.269 |                   202 |                 69      |                78 |                      0 |          54.1  |
|                4 | 341843     | 341843 (2008 EV5)        | NAO_ESPECTROGRAFADO |             6.291 |                   362 |                400      |              1486 |                      0 |          51.52 |
|                5 | 2006 BZ147 | (2006 BZ147)             | NAO_ESPECTROGRAFADO |             4.091 |                   362 |                 30.1662 |                64 |                      3 |          50.56 |
|                6 | 2024 BF    | (2024 BF)                | NAO_ESPECTROGRAFADO |             4.546 |                   362 |                 46.1952 |                82 |                      2 |          46.42 |
|                7 | 478784     | 478784 (2012 UV136)      | NAO_ESPECTROGRAFADO |             4.67  |                   394 |                 27.6948 |               148 |                      0 |          46.39 |
|                8 | 2020 MU1   | (2020 MU1)               | NAO_ESPECTROGRAFADO |             4.403 |                   370 |                 40.1871 |                54 |                      2 |          45.58 |
|                9 | 2023 KX3   | (2023 KX3)               | NAO_ESPECTROGRAFADO |             4.17  |                   370 |                 25.6905 |                80 |                      3 |          45.25 |
|               10 | 613862     | 613862 (2007 UY1)        | NAO_ESPECTROGRAFADO |             5.579 |                   362 |                 88.6228 |               312 |                      0 |          44.37 |

### 4.4 Veículos Lançadores Qualificados (Critério Atuarial Wilson 95% & Atividade NewSpace)
| modelo_veiculo   | agencia_principal   |   total_lancamentos |   taxa_sucesso_pct |   wilson_lower_95_pct |   ultimo_voo_ano |
|:-----------------|:--------------------|--------------------:|-------------------:|----------------------:|-----------------:|
| Falcon 9         | SPX                 |                 699 |              99.57 |                 98.75 |             2026 |
| Chang Zheng 2D   | SAST                |                 102 |              99.02 |                 94.65 |             2026 |
| Chang Zheng 2C   | CALT                |                  74 |              98.65 |                 92.73 |             2026 |
| Soyuz-FG         | RAKA                |                  70 |              98.57 |                 92.34 |             2019 |
| Chang Zheng 4B   | SAST                |                  59 |              98.31 |                 91    |             2026 |

### 4.5 Cenários Integrados de Subscrição Atuarial (Alvo $\times$ Lançador $\times$ Corredor Cislunar)
| id_cenario   | nome_asteroide           | modelo_veiculo   |   min_delta_v_kms |   indice_maturidade_info |   score_topsis |   score_risco_combinado | rating_atuarial   |
|:-------------|:-------------------------|:-----------------|------------------:|-------------------------:|---------------:|------------------------:|:------------------|
| NS_SCEN_001  | 99942 Apophis (2004 MN4) | Falcon 9         |             6.049 |                    98.89 |          60.62 |                   24.07 | A                 |
| NS_SCEN_002  | 99942 Apophis (2004 MN4) | Chang Zheng 2D   |             6.049 |                    98.89 |          60.62 |                   31.24 | BB                |
| NS_SCEN_003  | 99942 Apophis (2004 MN4) | Chang Zheng 2C   |             6.049 |                    98.89 |          60.62 |                   34.6  | BB                |
| NS_SCEN_004  | (2009 HC)                | Falcon 9         |             4.44  |                    69.67 |          57.25 |                   29.59 | A                 |
| NS_SCEN_005  | (2009 HC)                | Chang Zheng 2D   |             4.44  |                    69.67 |          57.25 |                   36.76 | BB                |
| NS_SCEN_006  | (2009 HC)                | Chang Zheng 2C   |             4.44  |                    69.67 |          57.25 |                   40.12 | B                 |
| NS_SCEN_007  | (2014 MF18)              | Falcon 9         |             6.269 |                    67.59 |          54.1  |                   29.46 | A                 |
| NS_SCEN_008  | (2014 MF18)              | Chang Zheng 2D   |             6.269 |                    67.59 |          54.1  |                   36.63 | BB                |

---

## 5. Plano Diretor de Recomendações para Tomada de Decisão

1. **Adoção de Arquitetura Híbrida de Dados (SQLite WAL + Parquet/DuckDB):** Manter o banco relacional indexado (`newspace_analytics.db`) como repositório transacional auditável (chaves estrangeiras ativas) acoplado a camadas colunares Parquet consultadas via DuckDB/Polars para sustentar o crescimento projetado até 2030 sem degradação de SLA no dashboard.
2. **Estratégia em Duas Fases para Prospecção Espacial:**
   - **Fase Alfa (Monetização de Informação Mineralógica):** Priorizar alvos no topo do ranking TOPSIS com $\Delta v \le 5.5\text{ km/s}$ e `OCC = 0` que ainda figuram como `NAO_ESPECTROGRAFADO`, comercializando dados proprietários de composição espectral e radar para agências e consórcios de recursos espaciais.
   - **Fase Beta (Missão de Extração / Demonstração ISRU):** Direcionar missões dedicadas exclusivamente a alvos já caracterizados nos complexos C (água/propelente orbital) e S/X (metais) com `OCC <= 1` e centenas de observações astrométricas.
3. **Diretriz de Subscrição Securitária Aeroespacial:** Restringir apólices de Rating `AA`/`A` a veículos lançadores ativos com limite inferior de confiabilidade de Wilson (95%) $\ge 95\%$, exigindo órbita de estacionamento inicial em `LEO_VLEO` ($< 450\text{ km}$) antes da queima de injeção interplanetária para minimizar tempo de exposição aos detritos de `LEO_SSO_ALTA` e `HEO_CISLUNAR`.
