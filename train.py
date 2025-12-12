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

data_not_na = data[~data[target].isna()]
data_not_na[target].isna().sum()
# %%

x = data_not_na[features.values()].copy()
y = data_not_na[target]

# converter idade para float64 para evitar problemas com valores ausentes no MLflow
x['idade'] = x['idade'].astype('float64')

from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# garantir que idade seja float64 em ambos os conjuntos
x_train['idade'] = x_train['idade'].astype('float64')
x_test['idade'] = x_test['idade'].astype('float64')


# %%

x_train.isna().sum()

from feature_engine import imputation
from feature_engine import encoding

input_classe = imputation.CategoricalImputer(
    fill_value="Não informado",
    variables=["ufOndeMora", 'cargoAtual', 'nivel',]
)

# OneHotEncoder é uma técnica usada para converter variáveis categóricas em variáveis binárias

onehot = encoding.OneHotEncoder(variables=['genero',
    'pcd',
    'ufOndeMora',
    'cargoAtual',
    'nivel',
    'tempoDeExperienciaEmDados',
    'tempoDeExperienciaEmTi',
])

from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics

clf = ensemble.GradientBoostingClassifier(n_estimators=500, learning_rate=0.6)

# cria um pipeline de ML 
# imputador -> encoder -> algoritmo
# imputador é responsável por preencher valores ausentes
# encoder converte variáveis categóricas em numéricas
# algoritmo treina o modelo de machine learning

modelo = pipeline.Pipeline(
    steps=[('imputador', input_classe),
           ('encoder', onehot),
           ("algoritmo", clf)]
)

import mlflow

# mlflow envia para o servidor
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_id=1)

# .fit é o que realmente chamam de machine learning
# treinar o modelo com rastreamento MLflow
with mlflow.start_run():
    mlflow.sklearn.autolog()
    
    modelo.fit(x_train, y_train)
    
    # fazer previsões no conjunto de treino
    y_train_pred = modelo.predict(x_train)
    
    # calcular acurácia no treino
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print(f"Acurácia no treino: {acc_train:.2%}")
    
    # fazer previsões no conjunto de teste
    y_test_pred = modelo.predict(x_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print(f"Acurácia no teste: {acc_test:.2%}")