# %% [markdown]
# # Aula 1 - Classificação: o que é e como funciona?

# %% [markdown]
# ## 1.1 - Apresentação
# 

# %% [markdown]
# ## 1.2 - Importando os dados

# %%
""" !pip install imblearn  """

# %%
""" !pip install seaborn """

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


