# %% [markdown]
# # Aula 1 - Classificação: o que é e como funciona?

# %% [markdown]
# ## 1.1 - Apresentação
# 

# %% [markdown]
# ## 1.2 - Importando os dados

# %%
""" pip install imblearn """

# %%
""" pip install seaborn """

# %%
import pandas as pd

# %%
dados = pd.read_csv('C:\Users\gabry\OneDrive\Área de Trabalho\ML Python\Classificação Pandas\Dados\Customer-Churn.csv')

# %%
dados.shape

# %%
dados.head()

# %% [markdown]
# ## 1.3 - Diferentes Variáveis

# %%
#modificação de forma manual 
traducao_dic = {'Sim': 1, 
                'Nao': 0}

dadosmodificados = dados[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(traducao_dic)
dadosmodificados.head()

# %%
#transformação pelo get_dummies
dummie_dados = pd.get_dummies(dados.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'],
                axis=1))

#junção dos dados trasformados com os que já tinhamos
dados_final = pd.concat([dadosmodificados, dummie_dados], axis=1)

# %%
dados_final.head()

# %% [markdown]
# ## 1.4 - Definição Informal
# 
# (slides)

# %% [markdown]
# ## 1.5 - Definição Formal

# %% [markdown]
# Informações para classificação:
# 
# $X$ = inputs (dados de entrada)
# 
# $y$ = outputs (dados de saída)

# %%
#DICA
pd.set_option('display.max_columns', 39)

# %%
dados_final.head()

# %% [markdown]
# 
# $y_i$ = $f(x_i)$

# %%
Xmaria = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]

# %%
#ymaria = ?

# %% [markdown]
# Novos pares de informações = ($Xmaria$, $ymaria$)

# %% [markdown]
# ## Balanceamento dos dados

# %%
#variável target está desbalanceada
import seaborn as sns
#matplotlib inline
ax = sns.countplot(x='Churn', data=dados_final)

# %%
dados_final.Churn.value_counts()

# %%
#biblioteca para balancear os dados utilizando over_sampling
from imblearn.over_sampling import SMOTE

# %%
#dividindo os dados em caracteristicas e target
X = dados_final.drop('Churn', axis = 1)
y = dados_final['Churn']

# %%
smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y)

# %%
#junção dos dados balanceados
dados_final = pd.concat([X, y], axis=1)

# %%
#verificação 1 - junção dos dados
dados_final.head(2)

# %%
#verificação 2 - balanceamento
ax = sns.countplot(x='Churn', data=dados_final)

# %%
dados_final.Churn.value_counts()

# %% [markdown]
# # Aula 2 - Método baseado na proximidade

# %% [markdown]
# ## 2.1 - Modelo K-nearest neighbors (KNN)
# 
# (Slides)

# %% [markdown]
# ## 2.2 - KNN por trás dos panos

# %%
Xmaria

# %%
#ymaria = ?

# %%
#Divisão em inputs e outputs
X = dados_final.drop('Churn', axis = 1)
y = dados_final['Churn']

# %%
#biblioteca para padronizar os dados
from sklearn.preprocessing import StandardScaler

# %%
norm = StandardScaler()

X_normalizado = norm.fit_transform(X)
X_normalizado

# %%
X_normalizado[0]

# %%
Xmaria_normalizado = norm.transform(pd.DataFrame(Xmaria, columns = X.columns))
Xmaria_normalizado

# %% [markdown]
# Distância Euclidiana:
# 
# $$\sqrt{\sum_{i=1}^k(a_{i}-b_{i})^2}$$

# %%
import numpy as np

# %%
a = Xmaria_normalizado

# %%
b = X_normalizado[0]

# %%
#1 - começamos subtraindo 
a - b

# %%
#2 - depois realizamos a exponenciação
np.square(a-b)

# %%
#3 - a soma 
np.sum(np.square(a-b))

# %%
#4 - então tiramos a raiz e temos nossa distância
np.sqrt(91.70603225977928)

# %% [markdown]
# ## 2.3 - Implementando o modelo

# %%
#biblioteca para divisão dos dados
from sklearn.model_selection import train_test_split

# %%
X_treino, X_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.3, random_state=123)

# %% [markdown]
# ### Treino e teste 

# %%
#biblioteca para criarmos o modelo de machine learning
from sklearn.neighbors import KNeighborsClassifier

# %%
#instanciar o modelo (criamos o modelo) - por padrão são 5 vizinhos  
knn = KNeighborsClassifier(metric='euclidean')

# %%
#treinando o modelo com os dados de treino
knn.fit(X_treino, y_treino)

# %%
#testando o modelo com os dados de teste
predito_knn = knn.predict(X_teste)

# %%
predito_knn

# %% [markdown]
# # Aula 3 - Método probabilístico

# %% [markdown]
# ## 3.1 - Teorema de Naive Bayes 
# 
# (slides)

# %% [markdown]
# ## 3.2 - Modelo Bernoulli Naive Bayes
# 
# (slides)

# %% [markdown]
# ## 3.3 - Treinamento e teste

# %%
#biblioteca para criarmos o modelo de machine learning
from sklearn.naive_bayes import BernoulliNB

# %%
#criamos o modelo
bnb = BernoulliNB(binarize=-0.44)

# %%
#escolho utilizar mediana, porque é o valor central dos nossos dados ordenados
np.median(X_treino)

# %%
X_treino

# %%
y_treino

# %%
#treinar o modelo
bnb.fit(X_treino, y_treino)

# %%
#testar o modelo
predito_BNb = bnb.predict(X_teste)

# %%
predito_BNb

# %% [markdown]
# # Aula 4 - Método Simbólico

# %% [markdown]
# ## 4.1 - O que é árvore de decisão?
# 
# (slides)

# %% [markdown]
# ## 4.2 - Por trás da árvore de decisão
# 
# (slides)

# %% [markdown]
# ## 4.3 - Implementando o modelo

# %%
#biblioteca para criarmos o modelo de machine learning
from sklearn.tree import DecisionTreeClassifier

# %%
#instanciando o modelo
dtc = DecisionTreeClassifier(criterion='entropy', random_state=42)

# %%
#treinar o modelo
dtc.fit(X_treino, y_treino)

# %%
#verificar a importância de cada atributo
dtc.feature_importances_

# %%
predito_ArvoreDecisao = dtc.predict(X_teste)

# %%
predito_ArvoreDecisao

# %% [markdown]
# # 5 - Validação dos modelos

# %% [markdown]
# ## 5.1 - Matriz de confusão
# 
# 

# %% [markdown]
# ![image.png](data:image/png;base64,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)

# %%
from sklearn.metrics import confusion_matrix

# %%
print(confusion_matrix(y_teste, predito_knn))

# %%
print(confusion_matrix(y_teste, predito_BNb))

# %%
print(confusion_matrix(y_teste, predito_ArvoreDecisao))

# %% [markdown]
# ## 5.2 - Acurácia
# A partir do calculo da matriz de confusão conseguimos inferir outras métricas, como por exemplo a acurária.
# 
# 
# 
# 
# $ACC$ = ${TP + TN \over TP + FP + TN + FN}$

# %%
from sklearn.metrics import accuracy_score

# %%
#modelo KNN
print(accuracy_score(y_teste, predito_knn))

# %%
#modelo Bernoulli de naive bayes
print(accuracy_score(y_teste, predito_BNb))

# %%
#modelo árvore de decisão
print(accuracy_score(y_teste, predito_ArvoreDecisao))

# %% [markdown]
# ## 5.3 - Precisão
# 
# Outra métrica importante é a precisão, que calcula quantos foram classificados corretamento como positivos ($TP$).
# 
# $PS$ = ${TP \over TP + FP}$

# %%
from sklearn.metrics import precision_score

# %%
#modelo KNN
print(precision_score(y_teste, predito_knn))

# %%
#modelo Bernoulli de naive bayes
print(precision_score(y_teste, predito_BNb))

# %%
#modelo árvore de decisão
print(precision_score(y_teste, predito_ArvoreDecisao))

# %% [markdown]
# ## 5.4 - Recall
# 
# Outra métrica importante é a Recall ou revocação ou ainda sensibilidade, que calcula o quão bom o modelo está para classificar corretamente um resultado positivo ($TP$).
# 
# $RC$ = ${TP \over TP + FN}$

# %%
from sklearn.metrics import recall_score

# %%
#modelo KNN
print(recall_score(y_teste, predito_knn))

# %%
#modelo Bernoulli de naive bayes
print(recall_score(y_teste, predito_BNb))

# %%
#modelo árvore de decisão
print(recall_score(y_teste, predito_ArvoreDecisao))

# %% [markdown]
# ## 5.5 - Escolhendo o melhor modelo
# 

# %%
#Exemplo - análise das precisões calculadas anteriormente
print('Modelo KNN: ', precision_score(y_teste, predito_knn))
print('Modelo Bernoulli de Naive Bayes: ', precision_score(y_teste, predito_BNb))
print('Modelo Árvore de Decisão: ', precision_score(y_teste, predito_ArvoreDecisao))


