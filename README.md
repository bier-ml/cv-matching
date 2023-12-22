# CV-Matching

## Demo

![Demo](data/demo1.gif)


## Technologies

- Python
- Hugginface infrastructure + Pytorch Lightning
- **TODO**

### Methodology

В качестве основы нашей работы мы используем векторное представление текстов (резюме/вакансии) с использованием
предобученных эмбедингов для русского языка. Мы обучаем модели машинного обучения, которые получают на вход
сконкатенированные векторы резюме и вакансии и на выходе предсказывают релевантность (вероятность) данного резюме к
данной вакансии.

В результате, обученная модель используется поиска релевантных вакансий, которых могло даже не быть в датасете. Также,
обученная модель позволяет выдавать релевантность конкретной вакансии к конкретному резюме.

## Data Preprocessing
To be done

## Training Pipeline

### Dataset

В качестве разбиения на `train` `test` `validation` была выбрана пропорция 80:10:10.
Из-за того, что датасет изначально
был сбалансирован проводилась случайная выборка без дополнительной стратификации. В процессе обучения, модель видела
только `train` множество, после каждой эпохи считались метрики на `validation` множестве, чтобы предотврадить
переобучение (где применимо) и обученная модель валидировалась на `test` множестве

### Metrics

В качестве метрик было выбрано 2 метрики: **Brier Score** и **ROC-AUC**. Первая метрика была выбрана из-за возможности
работы с непрерывными значениями вероятностей без необходимостей приведения предсказаний модели к логитам. Вторая
метрика, использовалась для оценки уверенности модели при подаче предсказаний.

| Metric      | Works with continuous predictions | Works with continuous True values score |
|-------------|-----------------------------------|-----------------------------------------|
| Brier Score | :white_check_mark:                | :white_check_mark:                      |
| ROC-AUC     | :white_check_mark:                | :x:                                     |

## Testing and Validation

### Experimental Setup

В качестве экспериментов были обучены и провалидированы декартово множество всех моделей и всех эмбедингов на всех
обозначенных выше метриках. Результат считался положительным, если при кросс-валидации на тестовом датасете метрика
превышала 0.5, так как
данный результат может
считаться не случайным.

### Experiments

| Model                     | Embedding                      | Brier score | ROC-AUC | 
|---------------------------|--------------------------------|-------------|---------|
| **Linear Regression**     | **RuBERT-tiny**                | 0.56        | 0.35    |
| **Linear Regression**     | **RuBERT**                     | 0.63        | 0.94    |
| **Linear Regression**     | **Slavic BERT**                | 0.36        | 0.47    |
| **Linear Regression**     | **Sentence RuBERT**            | 0.3         | 0.36    |
| **Linear Regression**     | **Sentence Multilingual BERT** | 0.91        | 0.71    |
| **CatBoost Regressor**    | **RuBERT-tiny**                | 0.69        | 0.17    |
| **CatBoost Regressor**    | **RuBERT**                     | 0.7         | 0.74    |
| **CatBoost Regressor**    | **Slavic BERT**                | 0.53        | 0.27    |
| **CatBoost Regressor**    | **Sentence RuBERT**            | 0.01        | 0.88    |
| **CatBoost Regressor**    | **Sentence Multilingual BERT** | 0.99        | 0.82    |
| **Two Layers Perceptron** | **RuBERT-tiny**                | 0.46        | 0.8     |
| **Two Layers Perceptron** | **RuBERT**                     | 0.55        | 0.65    |
| **Two Layers Perceptron** | **Slavic BERT**                | 0.34        | 0.99    |
| **Two Layers Perceptron** | **Sentence RuBERT**            | 0.86        | 0.79    |
| **Two Layers Perceptron** | **Sentence Multilingual BERT** | 0.73        | 0.46    |

## Deployment

Все модели были реализованы с использованием абстрактных интерфейсов, что позволяет единообразно использовать их в
сервисе и при необходимости добавлять новые модели. Взаимодействие моделей с пользователем происходит через streamlit
web-service. При загрузке своего резюме пользователю предлагается список из 5 самых релевантных вакансий.

## Product Details

### Context

To be done

### Interface

To be done

### Scaling

To be done