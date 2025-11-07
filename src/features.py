import numpy as np

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