# Databricks notebook source
# MAGIC %md
# MAGIC # Configurações de Pacotes

# COMMAND ----------

import pycaret

import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC # Configurações de Ambiente

# COMMAND ----------

# DBTITLE 1,Instância Widgets
# Name é para o identificador interno, usando no get, label é apenas visual
dbutils.widgets.text(name = 'input_path', defaultValue = '', label = 'Input path (Deve ser uma tabela delta)') 
dbutils.widgets.text(name = 'col_date', defaultValue = 'dt_ref', label = 'Coluna de Safra') 
dbutils.widgets.text(name = 'col_amostra', defaultValue = 'amostra', label = 'Coluna de amostra (DES, OOT, ESTAB, SIMUL)') 
dbutils.widgets.text(name = 'col_target', defaultValue = 'target', label = 'Coluna Target') 

input_path   = dbutils.widgets.get('input_path')
col_date     = dbutils.widgets.get('col_date')
col_amostra  = dbutils.widgets.get('col_amostra')
col_target   = dbutils.widgets.get('col_target')

# Variável para definir o tipo de problema
dbutils.widgets.dropdown(name = 'problem_type', defaultValue = 'regression', choices = ['classification', 'regression'], label = 'Tipo de Problema')

problem_type = dbutils.widgets.get('problem_type')

# COMMAND ----------

# MAGIC %md
# MAGIC # Carregando bases de dados

# COMMAND ----------

tb_input_ml = spark.read.table(
    input_path
)

display(tb_input_ml)

# COMMAND ----------

# MAGIC %md
# MAGIC # Preparando base para modelagem

# COMMAND ----------

X_des = (
    tb_input_ml
    .filter(F.col(col_amostra) == 'DES')
    .toPandas()
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Configurando PyCaret & Ajustando Modelos

# COMMAND ----------

from pycaret.regression import *

exp = setup(
    # ── dados ──────────────────────────────────────────────────────
    data          = X_des,
    target        = "target",

    # ── validação ──────────────────────────────────────────────────
    fold_strategy = "kfold",   # 'kfold', 'stratifiedkfold', 'timeseries'
    fold          = 5,         # número de folds
    fold_shuffle  = True,

    # ── pré-processamento ──────────────────────────────────────────
    normalize          = False,              # StandardScaler por padrão
    # normalize_method   = "zscore",          # 'zscore', 'minmax', 'maxabs', 'robust'
    transformation     = False,             # transformação de distribuição (Box-Cox, Yeo-Johnson)
    # handle_unknown_categorical = True,
    imputation_type    = "simple",          # 'simple' ou 'iterative'
    numeric_imputation = "median",          # 'mean', 'median', 'zero', 'mode'

    # ── MLflow ───────────────────────────────────────────────────── (apenas para futuro)
    # log_experiment  = True,
    # experiment_name = "/Shared/Yggdrasil/lgd_pycaret",
    # log_plots       = True,
    # log_data        = False,

    # ── outros ─────────────────────────────────────────────────────
    session_id      = 42,      # seed global para reprodutibilidade
    n_jobs          = -1,      # usa todos os cores disponíveis
)

# COMMAND ----------

best_model = compare_models(
    include = [          # lista explícita — recomendado para LGD
        "lr",            # Linear Regression
        "ridge",         # Ridge
        "lasso",         # Lasso
        "en",            # ElasticNet
        "rf",            # Random Forest
        "gbr",           # Gradient Boosting Regressor
        "lightgbm",      # LightGBM
        "et",            # Extra Trees
    ],
    sort="RMSE", 
    n_select = 3
)

# Captura a tabela exibida pelo compare_models
models_metrics = pull()

# COMMAND ----------

# tuned_top_model = tune_model(
#     best_model[0],
#     optimize = "RMSE",
#     search_library = "optuna",
#     search_algorithm = "tpe",
#     n_iter = 10
# )

# COMMAND ----------

# # Retreina o modelo com todos os dados (treino + validação) gerando o modelo final para produção
# tuned_top_model = finalize_model(tuned_top_model)

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.tree import DecisionTreeRegressor

# ════════════════════════════════════════════════════════════════
# PRÉ-REQUISITO: tb_input_ml_pred já tem a coluna "prediction" (score)
# ════════════════════════════════════════════════════════════════
pdf = tb_input_ml_pred.toPandas()

SCORE_COL  = "prediction"
TARGET_COL = "target"
SAMPLE_COL = "amostra"
ALPHA      = 0.05

# ── Função auxiliar: índice numérico → letra (0→A, 1→B, ...) ─────
def idx_para_letra(idx):
    """0→A, 1→B, ..., 25→Z, 26→AA, 27→AB, ..."""
    letras = ""
    idx += 1
    while idx > 0:
        idx, resto = divmod(idx - 1, 26)
        letras = chr(65 + resto) + letras
    return letras

# ════════════════════════════════════════════════════════════════
# FUNÇÃO DE FUSÃO POR INVERSÃO NO OOT (Mann-Whitney U)
# Recebe a coluna de grupos brutos (ordenados por LGD crescente)
# e devolve um mapeamento {grupo_raw → letra}
# ════════════════════════════════════════════════════════════════
def fundir_por_inversao(pdf, grupo_col, nome_rating):
    oot = pdf[pdf[SAMPLE_COL] == "OOT"]
    grupos_ordenados = sorted(pdf[grupo_col].unique())

    def lgd_medio_oot(grupo_list):
        mask = oot[grupo_col].isin(grupo_list)
        return oot.loc[mask, TARGET_COL].mean()

    def valores_oot(grupo_list):
        mask = oot[grupo_col].isin(grupo_list)
        return oot.loc[mask, TARGET_COL].values

    clusters = [[g] for g in grupos_ordenados]

    fundiu = True
    while fundiu:
        fundiu = False
        i = 1
        while i < len(clusters):
            prev_cluster = clusters[i - 1]
            curr_cluster = clusters[i]

            lgd_prev = lgd_medio_oot(prev_cluster)
            lgd_curr = lgd_medio_oot(curr_cluster)

            # Inversão: grupo de score maior tem LGD MENOR no OOT
            if pd.notna(lgd_prev) and pd.notna(lgd_curr) and lgd_curr < lgd_prev:
                v_prev = valores_oot(prev_cluster)
                v_curr = valores_oot(curr_cluster)

                if len(v_prev) > 0 and len(v_curr) > 0:
                    try:
                        stat, p_valor = mannwhitneyu(
                            v_prev, v_curr, alternative="two-sided"
                        )
                    except ValueError:
                        p_valor = 1.0

                    if p_valor > ALPHA:
                        clusters[i - 1] = prev_cluster + curr_cluster
                        clusters.pop(i)
                        fundiu = True
                        print(f"  [{nome_rating}] Fusão: {prev_cluster} + {curr_cluster} "
                              f"(inversão, p={p_valor:.4f} > {ALPHA})")
                        continue
                    else:
                        print(f"  [{nome_rating}] ATENÇÃO: inversão significativa entre "
                              f"{prev_cluster} e {curr_cluster} (p={p_valor:.4f}) — não fundido")
            i += 1

    # Mapeia raw → letra. Cluster 0 = menor LGD = A
    mapa = {}
    for novo_rating, cluster in enumerate(clusters):
        letra = idx_para_letra(novo_rating)
        for grupo_raw in cluster:
            mapa[grupo_raw] = letra

    print(f"  [{nome_rating}] Grupos finais após fusão: {len(clusters)}")
    return mapa

# ════════════════════════════════════════════════════════════════
# RATING 1 — Quantis de 0.05 + fusão por inversão no OOT
# ════════════════════════════════════════════════════════════════
des = pdf[pdf[SAMPLE_COL] == "DES"].copy()

quantis = np.arange(0.0, 1.0001, 0.05)
cortes  = des[SCORE_COL].quantile(quantis).values
cortes  = np.unique(cortes)
cortes[0]  = -np.inf
cortes[-1] =  np.inf

n_grupos_inicial = len(cortes) - 1
print(f"[RATING 1] Grupos iniciais por quantil: {n_grupos_inicial}")

pdf["rating_quantil_raw"] = pd.cut(
    pdf[SCORE_COL],
    bins=cortes,
    labels=range(n_grupos_inicial),
    include_lowest=True
).astype(int)

mapa_rating1 = fundir_por_inversao(pdf, "rating_quantil_raw", "RATING 1")
pdf["rating_quantil"] = pdf["rating_quantil_raw"].map(mapa_rating1)

# ════════════════════════════════════════════════════════════════
# RATING 2 — Árvore de regressão (score → target) + fusão por inversão
# ════════════════════════════════════════════════════════════════
X_des = des[[SCORE_COL]].values
y_des = des[TARGET_COL].values

tree = DecisionTreeRegressor(
    max_leaf_nodes = 10,
    min_samples_leaf = max(int(len(des) * 0.05), 50),
    random_state = 42
)
tree.fit(X_des, y_des)

n_folhas = tree.get_n_leaves()
print(f"\n[RATING 2] Folhas da árvore: {n_folhas}")

# Ordena folhas pelo LGD médio no DES e reindexa de 0 em diante (crescente)
folhas_all = tree.apply(pdf[[SCORE_COL]].values)
folha_lgd = (
    pd.DataFrame({"folha": tree.apply(X_des), "target": y_des})
    .groupby("folha")["target"].mean()
    .sort_values()
)
mapa_folha_ord = {folha: rank for rank, folha in enumerate(folha_lgd.index)}

# Grupo bruto da árvore: rank ordenado por LGD crescente (mesmo formato do quantil)
pdf["rating_arvore_raw"] = pd.Series(folhas_all, index=pdf.index).map(mapa_folha_ord)

# Aplica a MESMA fusão por inversão no OOT
mapa_rating2 = fundir_por_inversao(pdf, "rating_arvore_raw", "RATING 2")
pdf["rating_arvore"] = pdf["rating_arvore_raw"].map(mapa_rating2)

# ════════════════════════════════════════════════════════════════
# VALIDAÇÃO — monotonicidade por amostra
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("VALIDAÇÃO DE MONOTONICIDADE (LGD médio por rating)")
print("="*60)

for rating_col in ["rating_quantil", "rating_arvore"]:
    print(f"\n── {rating_col} ──")
    tabela = (
        pdf.groupby([rating_col, SAMPLE_COL])[TARGET_COL]
        .mean().unstack(SAMPLE_COL)
        .reindex(columns=["DES", "OOT", "ESTB"])
        .sort_index()
        .round(4)
    )
    print(tabela.to_string())

# ── limpa colunas auxiliares ─────────────────────────────────────
pdf = pdf.drop(columns=["rating_quantil_raw", "rating_arvore_raw"])

# ════════════════════════════════════════════════════════════════
# CONVERTE DE VOLTA PARA SPARK
# ════════════════════════════════════════════════════════════════
tb_input_ml_rating = spark.createDataFrame(pdf)
display(tb_input_ml_rating)

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ════════════════════════════════════════════════════════════════
# Carrega a tabela com ratings (no Databricks)
# ════════════════════════════════════════════════════════════════
pdf = tb_input_ml_rating.toPandas()

SCORE_COL  = "prediction"
TARGET_COL = "target"
SAMPLE_COL = "amostra"
DATE_COL   = "dt_ref"

pdf[DATE_COL] = pd.to_datetime(pdf[DATE_COL])
pdf["mes_ref"] = pdf[DATE_COL].dt.to_period("M").dt.to_timestamp()

sns.set_theme(style="whitegrid", context="notebook")
COR_BARRA, COR_LINHA, COR_VOL, PALETTE = "#4C72B0", "#C44E52", "#55A868", "viridis"

# ── Métricas globais avaliadas no OOT ────────────────────────────
def calc_metricas(df_sub):
    y, yhat = df_sub[TARGET_COL].values, df_sub[SCORE_COL].clip(0, 1).values
    mask = y > 0.01  # MAPE robusto: ignora targets ~0 que inflam a métrica
    mape = np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100
    return {
        "RMSE": mean_squared_error(y, yhat) ** 0.5,
        "MAE" : mean_absolute_error(y, yhat),
        "R2"  : r2_score(y, yhat),
        "MAPE": mape,
    }

met = calc_metricas(pdf[pdf[SAMPLE_COL] == "OOT"])

# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 16.5))
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.5, 1, 1],
                       hspace=0.55, wspace=0.30)

fig.suptitle("Dashboard de Modelo LGD  —  Yggdrasil", fontsize=22, fontweight="bold", y=0.995)
fig.text(0.5, 0.945, "Métricas avaliadas na amostra OOT", ha="center",
         fontsize=12, style="italic", color="#666")

# ── LINHA 1 — Cards de métrica ───────────────────────────────────
cards = [
    ("RMSE", met["RMSE"], "{:.4f}", COR_BARRA),
    ("MAE",  met["MAE"],  "{:.4f}", COR_VOL),
    ("R²",   met["R2"],   "{:.4f}", "#8172B3"),
    ("MAPE", met["MAPE"], "{:.1f}%", COR_LINHA),
]
for j, (nome, val, fmt, cor) in enumerate(cards):
    ax = fig.add_subplot(gs[0, j])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.04, 0.05), 0.92, 0.85, facecolor=cor, alpha=0.12,
                               edgecolor=cor, linewidth=2, transform=ax.transAxes))
    ax.text(0.5, 0.64, nome, ha="center", va="center", fontsize=17,
            fontweight="bold", color=cor, transform=ax.transAxes)
    ax.text(0.5, 0.30, fmt.format(val), ha="center", va="center", fontsize=29,
            fontweight="bold", color="#2c2c2a", transform=ax.transAxes)

# ── Função que desenha uma linha de rating (4 gráficos) ──────────
def linha_rating(row, rating_col, titulo):
    ratings = sorted(pdf[rating_col].dropna().unique())

    agg = (pdf.groupby(rating_col)
              .agg(target_medio=(TARGET_COL, "mean"), volume=(TARGET_COL, "size"))
              .reindex(ratings))
    agg["pct_vol"] = 100 * agg["volume"] / agg["volume"].sum()

    # G1 — barras LGD médio + linha de volumetria
    ax1 = fig.add_subplot(gs[row, 0])
    bars = ax1.bar(agg.index, agg["target_medio"], color=COR_BARRA, alpha=0.85,
                   edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, agg["target_medio"]):
        ax1.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.2f}", ha="center",
                 va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("LGD médio observado", fontsize=11, color=COR_BARRA, fontweight="bold")
    ax1.set_xlabel("Rating", fontsize=11)
    ax1.set_ylim(0, max(agg["target_medio"])*1.28)
    ax1.tick_params(axis="y", labelcolor=COR_BARRA)
    ax1.set_title(f"{titulo}  ·  LGD médio e volumetria por rating", fontsize=12.5, fontweight="bold")
    ax1b = ax1.twinx()
    ax1b.plot(agg.index, agg["pct_vol"], color=COR_VOL, marker="o", markersize=7, linewidth=2.5)
    ax1b.set_ylabel("% volumetria", fontsize=11, color=COR_VOL, fontweight="bold")
    ax1b.tick_params(axis="y", labelcolor=COR_VOL)
    ax1b.grid(False); ax1b.set_ylim(0, max(agg["pct_vol"])*1.3)

    # G2 — distribuição % por amostra
    ax2 = fig.add_subplot(gs[row, 1])
    vol = (pdf.groupby([rating_col, SAMPLE_COL]).size().unstack(SAMPLE_COL)
              .reindex(ratings).reindex(columns=["DES","OOT","ESTB"]))
    (vol.div(vol.sum(axis=0), axis=1)*100).plot(kind="bar", ax=ax2, width=0.78,
              color=["#4C72B0","#DD8452","#55A868"], edgecolor="white")
    ax2.set_title(f"{titulo}  ·  Distribuição (%) por amostra", fontsize=12.5, fontweight="bold")
    ax2.set_ylabel("% dentro da amostra", fontsize=11); ax2.set_xlabel("Rating", fontsize=11)
    ax2.legend(title="Amostra", fontsize=9); ax2.tick_params(axis="x", rotation=0)

    # G3 — série temporal LGD médio por rating
    ax3 = fig.add_subplot(gs[row, 2])
    serie = (pdf.groupby(["mes_ref", rating_col])[TARGET_COL].mean()
                .unstack(rating_col).reindex(columns=ratings))
    cores = sns.color_palette(PALETTE, len(ratings))
    for cor, rt in zip(cores, ratings):
        ax3.plot(serie.index, serie[rt], marker="o", markersize=3.5, linewidth=1.8, color=cor, label=rt)
    ax3.set_title(f"{titulo}  ·  LGD médio por rating no tempo", fontsize=12.5, fontweight="bold")
    ax3.set_ylabel("LGD médio", fontsize=11); ax3.set_xlabel("Mês de referência", fontsize=11)
    ax3.legend(title="Rating", fontsize=8, ncol=2, loc="upper left")
    ax3.tick_params(axis="x", rotation=45)

    # G4 — boxplot da dispersão do LGD por rating
    ax4 = fig.add_subplot(gs[row, 3])
    sns.boxplot(data=pdf, x=rating_col, y=TARGET_COL, order=ratings, hue=rating_col,
                palette=PALETTE, legend=False, ax=ax4, fliersize=1, linewidth=1)
    ax4.set_title(f"{titulo}  ·  Dispersão do LGD por rating", fontsize=12.5, fontweight="bold")
    ax4.set_ylabel("LGD observado", fontsize=11); ax4.set_xlabel("Rating", fontsize=11)

# ── Linha 2: árvore | Linha 3: quantil ──
linha_rating(1, "rating_arvore",  "Rating Árvore")
linha_rating(2, "rating_quantil", "Rating Quantil")

plt.show()