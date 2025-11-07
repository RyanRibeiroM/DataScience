import pandas as pd
def impute_missing_categorical(data):
    """
    Preenche valores ausentes (NaN) em colunas catégoricas especificas do dataset
    de acidentes com string 'Desconhecido'.

    Args:
        data (pd.DataFrame): DataFrame de entrada

    Returns:
        pd.DataFrame: O DataFrame com as colunas tratadas
    """

    data_cleaned = data.assign(
        regional = data['regional'].fillna('Desconhecido'),
        uop = data['uop'].fillna('Desconhecido'),
        delegacia = data['delegacia'].fillna('Desconhecido'),
        classificacao_acidente = data['classificacao_acidente'].fillna('Desconhecido')
    )

    return data_cleaned

def standardize_text_content(data):
    """
    Padroniza o conteúdo de todas as colunas de texto (object):
    1. Converte para minúsculo
    2. Remove espaços em branco no início e fim
    """
    df_cleaned = data.copy()
    
    colunas_de_texto = df_cleaned.select_dtypes(include=['object']).columns
    
    for col in colunas_de_texto:
        df_cleaned[col] = df_cleaned[col].str.lower().str.strip()
        
    return df_cleaned

def convert_data_types(data):
    """
    Converte colunas específicas do DataFrame para os tipos corretos.
    - 'data_inversa' para datetime
    - 'km' de string (com vírgula) para float

    Args:
        data (pd.DataFrame): O DataFrame de entrada

    Returns:
        pd.DataFrame: O DataFrame com os tipos de dados corrigidos
    """

    data_cleaned = data.assign(
        data_inversa = pd.to_datetime(data['data_inversa']),
        km = lambda x: x['km'].astype(str).str.replace(',', '.', regex=False).astype(float)
    )

    return data_cleaned