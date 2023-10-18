# %% [markdown]
# # Aula 1 - Classificação: o que é e como funciona?

# %% [markdown]
# ## 1.1 - Apresentação
# 

# %% [markdown]
# ## 1.2 - Importando os dados

# %%
import pandas as pd

# %%
dados = pd.read_csv('/content/sample_data/Customer-Churn.csv')

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


