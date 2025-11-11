import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

sns.set_theme(style="darkgrid")

def plot_principais_acidentes(df, coluna_causa="causa_acidente", coluna_mortos="mortos", top_n=5, titulo="Top Causas de Acidentes por Vítmas Fatais"):
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
    
def plot_tendencias(df, coluna_data="data_inversa", coluna_mes="mes_acidente", titulo="Tendência Mensal de Acidentes", periodo="Jan-Jul 2025"):
    
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
        print("⚙️  Coluna 'tipo_dia' criada automaticamente.")

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