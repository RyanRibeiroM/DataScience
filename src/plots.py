import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import math
from sklearn.metrics import confusion_matrix, classification_report

sns.set_theme(style="darkgrid")

def plot_principais_acidentes(df, coluna_causa="causa_acidente", coluna_mortos="mortos", top_n=5, titulo="Top Causas de Acidentes por Vítimas Fatais"):
    mortos_por_causa = (
        df.groupby(coluna_causa)[coluna_mortos].sum().reset_index().sort_values(coluna_mortos, ascending=False).head(top_n)
    )
    
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(
        data=mortos_por_causa,
        x=coluna_mortos,
        y=coluna_causa,
        ax=ax,
        color='red'
    )
    
    ax.set_title(f"{titulo} ({top_n} principais)")
    ax.set_xlabel("Total Vítimas Fatais")
    ax.set_ylabel("Causa do Acidente")


    plt.tight_layout()
    plt.show()


def plot_por_uf(df, coluna_uf="uf", top_n=10, titulo="Top 10 UFs com maiores registros de acidentes", periodo = "Jan-Ago 2025"):
    acidentes_uf = (
        df.groupby(coluna_uf)
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="acidentes")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=acidentes_uf,
        x=coluna_uf,
        y="acidentes",
        hue=coluna_uf,   
        palette="Blues_r",
        legend=False,
        ax=ax
    )

    ax.set_title(f"{titulo} ({periodo})", fontsize=12, fontweight="bold")
    ax.set_xlabel("UF (Estado)")
    ax.set_ylabel("Número de Acidentes")
    plt.tight_layout()
    plt.show()
    
def plot_tendencias(df, coluna_data="data_inversa", coluna_mes="mes_acidente", titulo="Tendência Mensal de Acidentes", periodo="Jan-Ago 2025"):
    
    if "mes_acidente" not in df.columns:
        if coluna_data not in df.columns:
            raise KeyError(f"A coluna '{coluna_data}' não existe no DataFrame.")
        df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce", dayfirst=True)
        df["mes_acidente"] = df[coluna_data].dt.month_name(locale="pt_BR").str.lower()
        
    ordem_dos_meses = [
        "janeiro", "fevereiro", "março", "abril",
        "maio", "junho", "julho", "agosto",
        "setembro", "outubro", "novembro", "dezembro"
    ]

    tendencia_acidentes = (
        df.groupby(coluna_mes)
        .size()
        .reset_index(name="acidentes")
    )

    tendencia_acidentes[coluna_mes] = pd.Categorical(
        tendencia_acidentes[coluna_mes],
        categories=ordem_dos_meses,
        ordered=True
    )

    tendencia_acidentes = tendencia_acidentes.sort_values(coluna_mes)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=tendencia_acidentes,
        x=coluna_mes,
        y="acidentes",
        marker='o',
        color='tab:blue',
        linewidth=2.5,
        ax=ax
    )

    ax.set_title(f"{titulo} ({periodo})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mês do Acidente")
    ax.set_ylabel("Número de Acidentes")

    plt.tight_layout()
    plt.show()
    
def plot_tendencia_dias(df, coluna_dia = "dia_semana", titulo="Tendência de Acidentes por Dia da Semana", destaque=("terça-feira", "quinta-feira")):
    if coluna_dia not in df.columns:
        raise KeyError(f"A coluna '{coluna_dia}' não existe no DataFrame.")

    tendencia = (
        df.groupby(coluna_dia)
        .size()
        .reset_index(name="acidentes")
    )

    ordem_dos_dias = [
        "segunda-feira", "terça-feira", "quarta-feira",
        "quinta-feira", "sexta-feira", "sábado", "domingo"
    ]

    tendencia[coluna_dia] = pd.Categorical(
        tendencia[coluna_dia],
        categories=ordem_dos_dias,
        ordered=True
    )

    tendencia = tendencia.sort_values(coluna_dia)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=tendencia,
        x=coluna_dia,
        y="acidentes",
        marker="o",
        linewidth=2.5,
        color="tab:blue",
        ax=ax
    )

    if destaque:
        ax.axvspan(destaque[0], destaque[1],
                   alpha=0.3,
                   color="orange",
                   label="Dias da semana com menos acidentes")

    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dia da Semana")
    ax.set_ylabel("Número de Acidentes")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_mortes_por_dia_e_tipo(df, coluna_tipo="tipo_acidente", coluna_mortos="mortos", coluna_dia="dia_semana", top_n=5, titulo="Total de Mortes Por Tipo de Acidente: Fim de Semana vs Dia de Semana"):
    if "tipo_dia" not in df.columns and "dia_semana" in df.columns:
        df["tipo_dia"] = df["dia_semana"].apply(
            lambda d: "Fim de Semana" if d in ["sábado", "domingo"] else "Dia de Semana"
        )

    top_tipos = df.groupby('tipo_acidente')['mortos'].sum().nlargest(top_n).index
    df_top = df[df['tipo_acidente'].isin(top_tipos)]

    g = sns.catplot(
        data=df_top,
        x='mortos',
        y='tipo_acidente',
        col='tipo_dia',
        kind='bar',
        estimator='sum',
        errorbar=None,
        height=5,
        aspect=1.2,
        order=top_tipos,
        color="Blue"
    )

    g.figure.suptitle(
        f"Total de Mortes por Tipo de Acidente — Fim de Semana vs. Dia de Semana (Top {top_n})",
        y=1.03, fontsize=12, fontweight="bold"
    )
    g.set_axis_labels("Total de Vítimas Fatais", "Tipo de Acidente")
    g.set_titles("Grupo: {col_name}")
    plt.tight_layout()
    plt.show()

def plot_model_performance(y_true, y_pred, labels, title="Performance do Modelo"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax_cm = axes[0]
    matrix = confusion_matrix(y_true, y_pred)

    im = ax_cm.imshow(matrix, cmap='PuBu')

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])

    l0 = labels[0] if len(labels) > 0 else '0'
    l1 = labels[1] if len(labels) > 1 else '1'
    
    ax_cm.set_xticklabels([f"{l0} (0)", f"{l1} (1)"], fontsize=12)
    ax_cm.set_yticklabels([f"{l0} (0)", f"{l1} (1)"], fontsize=12)
    
    ax_cm.set_xlabel("Predito", fontsize=13, fontweight='bold')
    ax_cm.set_ylabel("Real", fontsize=13, fontweight='bold')
    ax_cm.set_title("Matriz de Confusão", fontsize=15, pad=20, fontweight='bold')
    
    ax_cm.grid(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cor_texto = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax_cm.text(j, i, matrix[i, j], ha="center", va="center", 
                       color=cor_texto, fontsize=14, fontweight='bold')
            
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_cr = axes[1]
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    df_report = pd.DataFrame(report_dict).transpose()

    accuracy = report_dict.get('accuracy', 0)

    if 'accuracy' in df_report.index:
        df_report = df_report.drop('accuracy')
    
    metrics_df = df_report.iloc[:, :-1]
    
    sns.heatmap(metrics_df, annot=True, cmap='PuBu', fmt='.2f', 
                vmin=0, vmax=1, cbar=True, ax=ax_cr)
    
    ax_cr.set_title(f"Relatório de Classificação\nAcurácia Global: {accuracy:.2%}", 
                    fontsize=14, fontweight='bold', pad=20)
    ax_cr.set_yticks(range(len(metrics_df)), metrics_df.index, rotation=0)

    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

def plot_grid_performance(grid, metric_label="F1-Score (Macro)"):
    results = pd.DataFrame(grid.cv_results_)
    
    results['Estratégia'] = results['param_sampler'].apply(
        lambda x: str(x).split('(')[0] if x is not None else 'Original (Sem Sampler)'
    )

    param_cols = [c for c in results.columns if c.startswith('param_clf__') and results[c].nunique() > 1]
    
    best_idx = results['mean_test_score'].idxmax()
    best_score = results.loc[best_idx, 'mean_test_score']
    best_std = results.loc[best_idx, 'std_test_score']
    best_strategy = results.loc[best_idx, 'Estratégia']
    best_params = grid.best_params_

    params_str = " | ".join([f"{k.replace('clf__', '').replace('param_', '')}: {v}" for k, v in best_params.items()])

    plt.figure(figsize=(12, 2.5))
    plt.axis('off')

    plt.text(0.5, 0.75, f"Melhor Resultado: {best_score:.4f}", 
             fontsize=22, fontweight='bold', color='#2c3e50', ha='center', va='center')
    
    plt.text(0.5, 0.55, f"(Margem de erro / Aura: ±{best_std:.4f})", 
             fontsize=12, color='#7f8c8d', ha='center', va='center')

    plt.text(0.5, 0.35, f"Estratégia: {best_strategy}", 
             fontsize=14, fontweight='bold', color='#2980b9', ha='center', va='center')
    
    plt.text(0.5, 0.15, f"Parâmetros: {params_str}", 
             fontsize=11, family='monospace', color='#333333', ha='center', va='center',
             bbox=dict(facecolor='#f0f0f0', edgecolor='none', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()

    n_params = len(param_cols)
    if n_params == 0: return

    n_cols = 2
    n_rows = math.ceil(n_params / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), sharey=True)
    axes_flat = axes.flatten()

    sns.set_theme(style="whitegrid", palette="viridis")

    for i, param in enumerate(param_cols):
        ax = axes_flat[i]
        eixo_x_nome = param.replace("param_clf__", "").title()

        if isinstance(results[param].iloc[0], tuple):
            results[param] = results[param].astype(str)
            
        sns.lineplot(
            data=results, x=param, y='mean_test_score',
            hue='Estratégia', style='Estratégia',
            markers=True, dashes=False, linewidth=2.5, markersize=9,
            ax=ax, legend=(i == 0) 
        )
        
        best_x_global = results.loc[best_idx, param]
        best_y_global = results.loc[best_idx, 'mean_test_score']
        
        if isinstance(best_x_global, tuple):
             best_x_global = str(best_x_global)

        ax.scatter(best_x_global, best_y_global, s=200, facecolors='none', edgecolors='red', linewidth=2, zorder=5)

        if any(x in param for x in ['C', 'alpha', 'gamma', 'learning_rate']):
            try:
                ax.set_xscale('log')
            except:
                pass
            
        ax.set_title(f"Impacto de {eixo_x_nome}", fontsize=14, fontweight='bold')
        ax.set_xlabel(eixo_x_nome, fontsize=12)
        
        if i % n_cols == 0:
            ax.set_ylabel(metric_label, fontsize=12)
        else:
            ax.set_ylabel("")

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if n_params > 0:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            axes_flat[0].get_legend().remove()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.92), ncol=3, title="Estratégia")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
