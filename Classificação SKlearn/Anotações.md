# Ajuste e Previsão: Fundamentos do Estimador em Scikit-learn

O aprendizado de máquina (ML) é uma técnica que permite que os computadores aprendam com dados e façam previsões ou tomem decisões com base nesse aprendizado. A biblioteca Scikit-learn oferece diversos algoritmos e modelos integrados para ML, chamados de "estimadores". Para usar um estimador, você precisa ajustá-lo aos seus dados.

## Exemplo de Uso do RandomForestClassifier

`from sklearn.ensemble import RandomForestClassifier`

*#Crie uma instância do estimador*

`clf = RandomForestClassifier(random_state=0)`

## *#Prepare seus dados de treinamento (X) e os rótulos (y)*

*#Ajuste o estimador aos dados de treinamento*

`clf.fit(X, y)`

Uma vez ajustado, o estimador pode ser usado para fazer previsões em novos dados sem a necessidade de treinamento adicional.

---



## Transformadores e Pré-processadores

Scikit-learn também fornece transformadores e pré-processadores para preparar os dados antes do treinamento. Estes seguem a mesma API dos estimadores, mas eles têm métodos de transformação em vez de previsão.

Exemplo de uso do StandardScaler:

```python
from sklearn.preprocessing import StandardScaler

# Prepare seus dados (X)

# Aplique uma transformação aos dados
StandardScaler().fit(X).transform(X)
```


## Pipelines: Encadeando Pré-processadores e Estimadores

Transformadores e estimadores (preditores) podem ser combinados em um único objeto chamado Pipeline. Isso ajuda a evitar vazamento de dados e simplifica o fluxo de trabalho.

Exemplo de uso do Pipeline:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Crie um pipeline com transformadores e um estimador final
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# Ajuste o pipeline aos dados de treinamento
pipe.fit(X_train, y_train)

# Use o pipeline para fazer previsões
accuracy_score(pipe.predict(X_test), y_test)
```


## Avaliação do Modelo

É importante avaliar o desempenho do modelo ajustado. Scikit-learn oferece ferramentas para validação cruzada e busca automática de melhores parâmetros, garantindo que seu modelo seja eficaz em dados não vistos.

Exemplo de uso da validação cruzada:

```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

# Realize um procedimento de validação cruzada
result = cross_validate(LinearRegression(), X, y)

# Obtenha os resultados da validação cruzada
result['test_score']
```


## Pesquisas Automáticas de Parâmetros

Scikit-learn fornece ferramentas para encontrar automaticamente as melhores combinações de parâmetros (por meio de validação cruzada), garantindo um melhor desempenho do modelo.

Exemplo de pesquisa de parâmetros com RandomizedSearchCV:

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Defina o espaço de parâmetros a ser pesquisado
param_distributions = {'n_estimators': randint(1, 5), 'max_depth': randint(5, 10)}

# Crie um objeto de pesquisa e ajuste-o aos dados
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                            n_iter=5,
                            param_distributions=param_distributions,
                            random_state=0)
search.fit(X_train, y_train)

# Obtenha os melhores parâmetros encontrados
search.best_params_
```
