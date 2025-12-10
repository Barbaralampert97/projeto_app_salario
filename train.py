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

# %%

# ver as classes da variável target
data[target].unique()


depara_salario = {
    'Menos de R$ 1.000/mês':'01 - Menos de R$ 1.000/mês',
    'de R$ 1.001/mês a R$ 2.000/mês':'02 - R$ 1.001/mês a R$ 2.000/mês',
    'de R$ 2.001/mês a R$ 3.000/mês':'03 - R$ 2.001/mês a R$ 3.000/mês',
    'de R$ 3.001/mês a R$ 4.000/mês':'04 - R$ 3.001/mês a R$ 4.000/mês',
    'de R$ 4.001/mês a R$ 6.000/mês':'05 - R$ 4.001/mês a R$ 6.000/mês',
    'de R$ 6.001/mês a R$ 8.000/mês':'06 - R$ 6.001/mês a R$ 8.000/mês',
    'de R$ 8.001/mês a R$ 12.000/mês':'07 - R$ 8.001/mês a R$ 12.000/mês',
    'de R$ 12.001/mês a R$ 16.000/mês':'08 - R$ 12.001/mês a R$ 16.000/mês',
    'de R$ 16.001/mês a R$ 20.000/mês':'09 - R$ 16.001/mês a R$ 20.000/mês',
    'de R$ 20.001/mês a R$ 25.000/mês':'10 - R$ 20.001/mês a R$ 25.000/mês',
    'de R$ 25.001/mês a R$ 30.000/mês':'11 - R$ 25.001/mês a R$ 30.000/mês',
    'de R$ 30.001/mês a R$ 40.000/mês':'12 - R$ 30.001/mês a R$ 40.000/mês',
    'Acima de R$ 40.001/mês':'13 - Acima de R$ 40.001/mês',
}

data[target] = data[target].replace(depara_salario)

data
# %%

x = data[features.values()]
y = data[target]
y
# %%
