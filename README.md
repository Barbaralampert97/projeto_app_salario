# 💰 Preditor de Salários - State of Data 2024

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-red.svg)
![MLflow](https://img.shields.io/badge/mlflow-tracking-green.svg)
![scikit-learn](https://img.shields.io/badge/sklearn-ML-orange.svg)

Uma aplicação de Machine Learning para predição de faixas salariais de profissionais da área de dados no Brasil, baseada no dataset **State of Data 2024 - Kaggle**.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#funcionalidades)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Pipeline de Machine Learning](#pipeline-de-machine-learning)
- [Modelagem](#modelagem)
- [Resultados](#resultados)
- [Interface da Aplicação](#interface-da-aplicação)
- [Contribuindo](#contribuindo)

## 🎯 Sobre o Projeto

Este projeto utiliza dados reais de profissionais da área de dados para criar um modelo preditivo que estima faixas salariais com base em características profissionais e demográficas. O objetivo é fornecer insights sobre o mercado de dados no Brasil e ajudar profissionais a entenderem melhor o cenário salarial da área.

### Contexto

O dataset utilizado é proveniente da pesquisa **State of Data 2024**, realizada pela comunidade Kaggle, e contém informações sobre:
- Perfil demográfico (idade, gênero, PCD)
- Localização (UF)
- Experiência profissional
- Cargo e nível hierárquico
- Faixa salarial

## ✨ Funcionalidades

- 🎯 **Predição de Salário**: Estimativa de faixa salarial baseada em perfil profissional
- 📊 **Interface Interativa**: Aplicação web intuitiva construída com Streamlit
- 🔄 **MLflow Integration**: Rastreamento completo de experimentos e versionamento de modelos
- 🤖 **Pipeline Automatizado**: Pré-processamento e treinamento automatizados
- 📈 **Múltiplas Faixas Salariais**: Predição entre 13 faixas salariais diferentes
- ⚡ **Cache Inteligente**: Carregamento otimizado de modelos com cache de 1 dia

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.8+**: Linguagem principal
- **Streamlit**: Framework para interface web
- **MLflow**: Gerenciamento de experimentos e modelos
- **scikit-learn**: Framework de Machine Learning

### Pré-processamento
- **pandas**: Manipulação de dados
- **feature-engine**: Engenharia de features avançada
  - `CategoricalImputer`: Imputação de valores faltantes
  - `OneHotEncoder`: Codificação de variáveis categóricas

### Modelo
- **GradientBoostingClassifier**: Algoritmo de classificação

### Utilitários
- **pyarrow**: Otimização de dataframes

## 📁 Estrutura do Projeto

```
projeto_app_salario/
│
├── app.py                          # Aplicação Streamlit
├── train.py                        # Script de treinamento
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação
├── arch.drawio                     # Diagrama de arquitetura
│
├── data/                           # Diretório de dados
│   ├── Final Dataset - State of Data 2024 - Kaggle - df_survey_2024.csv
│   └── template.csv                # Template com features do treino
│
├── mlartifacts/                    # Artefatos do MLflow
│   └── 1/                          # Experiment ID 1
│       ├── [run_ids]/              # Runs individuais
│       │   └── artifacts/          # Métricas e visualizações
│       └── models/                 # Modelos salvos
│
└── mlruns/                         # Metadata do MLflow
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip
- Ambiente virtual (recomendado)

### Passo a Passo

1. **Clone o repositório**
```powershell
git clone https://github.com/Barbaralampert97/projeto_app_salario.git
cd projeto_app_salario
```

2. **Crie um ambiente virtual**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. **Instale as dependências**
```powershell
pip install -r requirements.txt
pip install mlflow scikit-learn feature-engine
```

4. **Baixe o dataset**
   - Coloque o arquivo CSV na pasta `data/`
   - Nome: `Final Dataset - State of Data 2024 - Kaggle - df_survey_2024.csv`

## 💻 Como Usar

### 1. Inicie o MLflow Server

```powershell
mlflow server --host 127.0.0.1 --port 5000
```

Acesse a interface do MLflow em: `http://127.0.0.1:5000`

### 2. Treine o Modelo

```powershell
python train.py
```

Este script irá:
- Carregar e processar os dados
- Criar o pipeline de pré-processamento
- Treinar o modelo GradientBoosting
- Registrar métricas no MLflow
- Salvar o modelo treinado

### 3. Execute a Aplicação

```powershell
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em: `http://localhost:8501`

## 🔄 Pipeline de Machine Learning

### 1. **Carregamento e Preparação dos Dados**

```python
# Features selecionadas
- idade
- genero
- pcd (Pessoa com Deficiência)
- ufOndeMora
- cargoAtual
- nivel
- tempoDeExperienciaEmDados
- tempoDeExperienciaEmTi

# Target
- faixa_salarial (13 faixas)
```

### 2. **Pré-processamento**

#### Imputação Categórica
```python
CategoricalImputer(
    fill_value="Não informado",
    variables=["ufOndeMora", "cargoAtual", "nivel"]
)
```

#### Codificação One-Hot
```python
OneHotEncoder(
    variables=[
        'genero', 'pcd', 'ufOndeMora', 'cargoAtual',
        'nivel', 'tempoDeExperienciaEmDados', 'tempoDeExperienciaEmTi'
    ]
)
```

### 3. **Modelo**

```python
GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.6
)
```

### 4. **Pipeline Completo**

```python
Pipeline([
    ('imputador', imput_classe),
    ('encoder', onehot),
    ('algoritmo', clf)
])
```

## 📊 Modelagem

### Faixas Salariais

| Código | Faixa Salarial |
|--------|----------------|
| 01 | Menos de R$ 1.000/mês |
| 02 | R$ 1.001 a R$ 2.000/mês |
| 03 | R$ 2.001 a R$ 3.000/mês |
| 04 | R$ 3.001 a R$ 4.000/mês |
| 05 | R$ 4.001 a R$ 6.000/mês |
| 06 | R$ 6.001 a R$ 8.000/mês |
| 07 | R$ 8.001 a R$ 12.000/mês |
| 08 | R$ 12.001 a R$ 16.000/mês |
| 09 | R$ 16.001 a R$ 20.000/mês |
| 10 | R$ 20.001 a R$ 25.000/mês |
| 11 | R$ 25.001 a R$ 30.000/mês |
| 12 | R$ 30.001 a R$ 40.000/mês |
| 13 | Acima de R$ 40.001/mês |

### Divisão dos Dados

- **Training Set**: 80% dos dados
- **Test Set**: 20% dos dados
- **Estratificação**: Por faixa salarial (target)
- **Random State**: 42 (reprodutibilidade)

### Métricas Avaliadas

- **Accuracy (Train)**: Acurácia no conjunto de treino
- **Accuracy (Test)**: Acurácia no conjunto de teste
- Métricas adicionais disponíveis no MLflow

## 📱 Interface da Aplicação

A aplicação Streamlit possui uma interface dividida em 3 colunas:

### Coluna 1: Dados Pessoais
- 📅 **Idade**: Seletor numérico
- 👤 **Gênero**: Lista suspensa
- ♿ **PCD**: Pessoa com Deficiência (Sim/Não)
- 📍 **UF**: Estado onde mora

### Coluna 2: Dados Profissionais
- 💼 **Cargo Atual**: Posição atual
- 📊 **Nível**: Hierárquico (Júnior, Pleno, Sênior, etc.)

### Coluna 3: Experiência
- 📈 **Tempo de Exp. em Dados**: Anos de experiência
- 💻 **Tempo de Exp. em TI**: Anos de experiência total

### Resultado
Após preencher os campos, a aplicação exibe:
```
Sua faixa salarial: R$ X.XXX a R$ Y.YYY/mês
```

## 📈 Resultados

Os resultados podem ser visualizados através do MLflow UI:

1. Acesse: `http://127.0.0.1:5000`
2. Navegue até o experimento ID 1
3. Compare diferentes runs
4. Visualize métricas e artefatos




## 👥 Autores

- **Barbara Lampert** - [@Barbaralampert97](https://github.com/Barbaralampert97)


