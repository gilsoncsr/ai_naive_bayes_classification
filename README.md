# Predição de Obesidade com Machine Learning

## Visão Geral

Este projeto utiliza técnicas de Análise Exploratória de Dados (EDA) e Aprendizado de Máquina para prever o risco de obesidade com base em dados sobre hábitos alimentares, atividade física e estilo de vida. A aplicação inclui um modelo treinado e uma API Flask para realizar previsões.

## Tecnologias Utilizadas

- **Linguagem**: Python 3
- **Bibliotecas**: Pandas, Scikit-Learn, Matplotlib, Plotly, Statsmodels, Sweetviz, Optuna, Flask, Pydantic
- **Gerenciamento de dependências**: Pipenv

## Instalação

Para instalar as dependências do projeto, execute:

```sh
pipenv install pandas plotly matplotlib statsmodels nbformat ipykernel scikit-learn ipywidgets sweetviz flask pydantic pyarrow Flask-Pydantic optuna joblib
```

## Uso

### 1. Análise Exploratória de Dados (EDA)

O projeto realiza uma análise visual e estatística do conjunto de dados, incluindo:

- Distribuição das variáveis
- Testes estatísticos para verificação de hipóteses
- Seleção de variáveis relevantes

### 2. Treinamento do Modelo

- O modelo utiliza **Naive Bayes** como baseline.
- A seleção de features é realizada com **SelectKBest e Chi-Square**.
- Hiperparâmetros são otimizados com **Optuna**.

### 3. API Flask para Predição

A API permite fazer previsões com o modelo treinado.

#### Inicializar a API

Para rodar a API, execute:

```sh
pipenv shell
python app.py
```

#### Fazer uma Predição

Envie uma requisição POST para `http://127.0.0.1:5000/predict` com um JSON no seguinte formato:

```json
{
  "Genero_Masculino": 1,
  "Idade": 25,
  "Historico_Familiar_Sobrepeso": 1,
  "Consumo_Alta_Caloria_Com_Frequencia": 2,
  "Consumo_Vegetais_Com_Frequencia": 3,
  "Refeicoes_Dia": 3,
  "Consumo_Alimentos_entre_Refeicoes": 2,
  "Fumante": 0,
  "Consumo_Agua": 3,
  "Monitora_Calorias_Ingeridas": 1,
  "Nivel_Atividade_Fisica": 2,
  "Nivel_Uso_Tela": 3,
  "Consumo_Alcool": 1,
  "Transporte_Automovel": 1,
  "Transporte_Bicicleta": 0,
  "Transporte_Motocicleta": 0,
  "Transporte_Publico": 1,
  "Transporte_Caminhada": 0
}
```

A API retorna a classificação de obesidade prevista pelo modelo.

## Resultados e Métricas

O modelo é avaliado com:

- **Relatório de Classificação** (Precision, Recall, F1-score)
- **Matriz de Confusão**
- **Otimização de Hiperparâmetros com Optuna**

---

# Obesity Prediction with Machine Learning

## Overview

This project utilizes Exploratory Data Analysis (EDA) and Machine Learning techniques to predict obesity risk based on data related to eating habits, physical activity, and lifestyle. The application includes a trained model and a Flask API for predictions.

## Technologies Used

- **Language**: Python 3
- **Libraries**: Pandas, Scikit-Learn, Matplotlib, Plotly, Statsmodels, Sweetviz, Optuna, Flask, Pydantic
- **Dependency Management**: Pipenv

## Installation

To install the project dependencies, run:

```sh
pipenv install pandas plotly matplotlib statsmodels nbformat ipykernel scikit-learn ipywidgets sweetviz flask pydantic pyarrow Flask-Pydantic optuna joblib
```

## Usage

### 1. Exploratory Data Analysis (EDA)

The project performs a visual and statistical analysis of the dataset, including:

- Variable distribution
- Statistical hypothesis testing
- Selection of relevant features

### 2. Model Training

- The model uses **Naive Bayes** as a baseline.
- Feature selection is performed using **SelectKBest and Chi-Square**.
- Hyperparameters are optimized with **Optuna**.

### 3. Flask API for Prediction

The API allows predictions using the trained model.

#### Start the API

To run the API, execute:

```sh
pipenv shell
python app.py
```

#### Make a Prediction

Send a POST request to `http://127.0.0.1:5000/predict` with a JSON in the following format:

```json
{
  "Genero_Masculino": 1,
  "Idade": 25,
  "Historico_Familiar_Sobrepeso": 1,
  "Consumo_Alta_Caloria_Com_Frequencia": 2,
  "Consumo_Vegetais_Com_Frequencia": 3,
  "Refeicoes_Dia": 3,
  "Consumo_Alimentos_entre_Refeicoes": 2,
  "Fumante": 0,
  "Consumo_Agua": 3,
  "Monitora_Calorias_Ingeridas": 1,
  "Nivel_Atividade_Fisica": 2,
  "Nivel_Uso_Tela": 3,
  "Consumo_Alcool": 1,
  "Transporte_Automovel": 1,
  "Transporte_Bicicleta": 0,
  "Transporte_Motocicleta": 0,
  "Transporte_Publico": 1,
  "Transporte_Caminhada": 0
}
```

The API returns the predicted obesity classification.

## Results and Metrics

The model is evaluated using:

- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**
- **Hyperparameter Optimization with Optuna**

---

# Predicción de Obesidad con Aprendizaje Automático

## Resumen

Este proyecto utiliza Análisis Exploratorio de Datos (EDA) y técnicas de Aprendizaje Automático para predecir el riesgo de obesidad en función de datos relacionados con hábitos alimenticios, actividad física y estilo de vida. La aplicación incluye un modelo entrenado y una API Flask para predicciones.

## Tecnologías Utilizadas

- **Lenguaje**: Python 3
- **Librerías**: Pandas, Scikit-Learn, Matplotlib, Plotly, Statsmodels, Sweetviz, Optuna, Flask, Pydantic
- **Gestor de Dependencias**: Pipenv

## Instalación

Para instalar las dependencias del proyecto, ejecuta:

```sh
pipenv install pandas plotly matplotlib statsmodels nbformat ipykernel scikit-learn ipywidgets sweetviz flask pydantic pyarrow Flask-Pydantic optuna joblib
```

## Uso

### 1. Análisis Exploratorio de Datos (EDA)

El proyecto realiza un análisis visual y estadístico del conjunto de datos, incluyendo:

- Distribución de variables
- Pruebas de hipótesis estadísticas
- Selección de características relevantes

### 2. Entrenamiento del Modelo

- Se utiliza **Naive Bayes** como modelo base.
- La selección de características se realiza mediante **SelectKBest y Chi-Cuadrado**.
- Los hiperparámetros se optimizan con **Optuna**.

### 3. API Flask para Predicción

La API permite realizar predicciones utilizando el modelo entrenado.

#### Iniciar la API

Para ejecutar la API, ejecuta:

```sh
pipenv shell
python app.py
```

#### Realizar una Predicción

Envía una solicitud POST a `http://127.0.0.1:5000/predict` con un JSON en el siguiente formato:

```json
{
  "Genero_Masculino": 1,
  "Idade": 25,
  "Historico_Familiar_Sobrepeso": 1,
  "Consumo_Alta_Caloria_Com_Frequencia": 2,
  "Consumo_Vegetais_Com_Frequencia": 3,
  "Refeicoes_Dia": 3,
  "Consumo_Alimentos_entre_Refeicoes": 2,
  "Fumante": 0,
  "Consumo_Agua": 3,
  "Monitora_Calorias_Ingeridas": 1,
  "Nivel_Atividade_Fisica": 2,
  "Nivel_Uso_Tela": 3,
  "Consumo_Alcool": 1,
  "Transporte_Automovel": 1,
  "Transporte_Bicicleta": 0,
  "Transporte_Motocicleta": 0,
  "Transporte_Publico": 1,
  "Transporte_Caminhada": 0
}
```

La API devuelve la clasificación de obesidad predicha.

## Resultados y Métricas

El modelo se evalúa utilizando:

- **Informe de Clasificación** (Precisión, Recall, F1-score)
- **Matriz de Confusión**
- **Optimización de Hiperparámetros con Optuna**
