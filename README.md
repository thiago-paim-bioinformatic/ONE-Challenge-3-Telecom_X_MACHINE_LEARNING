# 📊 Previsão de Evasão de Clientes (Churn) da TelecomX - Modelos de Machine Learning

## 🎯 Objetivo

Este projeto tem como foco a análise de evasão de clientes (churn) da empresa **TelecomX**, utilizando dados históricos para identificar padrões comportamentais e contratuais que influenciam a saída dos clientes. A partir dessa análise, o objetivo é construir modelos preditivos capazes de antecipar o risco de churn, contribuindo para estratégias de retenção mais eficazes.

---

## 🧠 Metodologia

A abordagem adotada seguiu um pipeline completo de ciência de dados, com as seguintes etapas:

### 1. Preparação e Normalização dos Dados

- As variáveis categóricas foram transformadas em binárias (0 ou 1).
- Variáveis contínuas como `customer.tenure`, `account.Charges.Monthly` e `account.Charges.Total` mantiveram suas escalas originais.
- Para garantir uniformidade, os dados foram normalizados entre 0 e 1 utilizando o método `MinMaxScaler` da biblioteca `sklearn.preprocessing`.

###  Balanceamento de Dados
  
Como a variável resposta (Churn) apresenta desbalanceamento entre classes, foram utilizadas técnicas para equilibrar os dados.

🔸 Oversampling

Aumenta a quantidade de dados da classe minoritária até igualar à classe majoritária. Dessa forma, o modelo aprende melhor os padrões da classe menos representada.
from imblearn.over_sampling import SMOTE


🔸 Undersampling

Reduz a quantidade de dados da classe majoritária até igualar à classe minoritária. Isso evita que o modelo se concentre excessivamente na classe dominante.
from imblearn.under_sampling import NearMiss



- Validação Cruzada com Pipeline

Foi utilizado o método StratifiedKFold para garantir que a proporção das classes seja mantida em cada divisão. A biblioteca imblearn permite automatizar o processo de validação cruzada com técnicas de oversampling ou undersampling integradas ao pipeline.
from sklearn.model_selection import StratifiedKFold


Essa abordagem garante que o balanceamento dos dados ocorra dentro de cada fold, evitando vazamento de dados e proporcionando uma avaliação mais robusta dos modelos.

🛠️ Ferramentas e Bibliotecas
- Python – Linguagem principal do projeto
- pandas – Manipulação de dados
- matplotlib, seaborn, plotly.express – Visualização gráfica
- scikit-learn – Modelos de machine learning, pré-processamento e validação
- imblearn – Técnicas de balanceamento e integração com pipelines
- MinMaxScaler – Normalização de variáveis contínuas


## Modelos de Classificação Utilizados

### 🔹 Modelo Base (Dummy)

O modelo base serve como referência para avaliar o desempenho dos demais algoritmos. Ele realiza previsões simples, como sempre escolher a classe majoritária, sem considerar os dados de entrada. É útil para estabelecer um ponto de partida mínimo de performance.

### 🔹 Árvore de Decisão

A Árvore de Decisão é um modelo interpretável que segmenta os dados em ramos com base em perguntas binárias sobre os atributos. A profundidade da árvore (`max_depth`) influencia diretamente sua capacidade de generalização: árvores muito profundas tendem a sobreajustar os dados, enquanto árvores rasas podem subestimar padrões importantes.

### 🔹 Random Forest

O Random Forest é um modelo de conjunto (ensemble) que combina várias árvores de decisão independentes. Cada árvore é treinada com uma amostra aleatória dos dados e dos atributos, e a decisão final é tomada por votação. Essa abordagem reduz o risco de sobreajuste e melhora a robustez do modelo.

### 🔹 KNN (K-Nearest Neighbors)

O KNN é um algoritmo baseado em instâncias que classifica um exemplo com base na maioria das classes dos seus vizinhos mais próximos. O parâmetro `n_neighbors` define quantos vizinhos serão considerados. O desempenho do KNN depende da escala dos dados e da escolha adequada de `n_neighbors`.

----
Este projeto representa uma aplicação prática de ciência de dados voltada à inteligência de negócios, com potencial direto de impacto na redução da evasão de clientes da TelecomX.
