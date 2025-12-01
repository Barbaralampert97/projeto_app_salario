# %%

import pandas as pd

data = pd.read_csv('data/Final Dataset - State of Data 2024 - Kaggle - df_survey_2024.csv')

data.head()

# %%
for i in data.columns:
    print(i)
 
#variáveis independentes   
features = {
    "1.a_idade": "idade",
    "1.b_genero": "genero",
    "1.d_pcd": "pcd",
    "1.i.1_uf_onde_mora": "ufOndeMora",
    "2.f_cargo_atual": "cargoAtual",
    "2.g_nivel": "nivel",
    "2.i_tempo_de_experiencia_em_dados": "tempoDeExperienciaEmDados",
    "2.j_tempo_de_experiencia_em_ti": "tempoDeExperienciaEmTi",
}
#variável dependente
target = "2.h_faixa_salarial"

# criando uma lista com todas as colunas que serão usadas
columns = list(features.keys()) + [target]


data = data[columns].copy()
data.rename(columns=features, inplace=True)

data 