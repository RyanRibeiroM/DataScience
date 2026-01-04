import numpy as np
from sklearn.preprocessing import StandardScaler

def Add_new_columns(data):
    """
    Cria uma nova coluna 'mes_acidente' com o nome do mês 
    extraído da 'data_inversa'.
    """
    month_map = {
        1: "janeiro",
        2: "fevereiro",
        3: "março",
        4: "abril",
        5: "maio",
        6: "junho",
        7: "julho",
        8: "agosto",
        9: "setembro",
        10: "outubro",
        11: "novembro",
        12: "dezembro"
    }

    month_numbers = data['data_inversa'].dt.month
    
    data['mes_acidente'] = month_numbers.map(month_map) 

    dias_fim_de_semana = ['sábado', 'domingo']


    data['tipo_dia'] = np.where(
        data['dia_semana'].isin(dias_fim_de_semana), 
        'Fim de Semana', 
        'Dia de Semana'
    )

    return data

def prepare_modeling_data_heart_failure(df):
    feature_cols = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    target_col = 'DEATH_EVENT'

    X = df[feature_cols].copy()
    Y = df[target_col]

    numerical_features = ['age','creatinine_phosphokinase','ejection_fraction', 'platelets','serum_creatinine', 'serum_sodium', 'time']
    X[numerical_features] = StandardScaler().fit_transform(X[numerical_features])

    return X, Y