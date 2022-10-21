#!/usr/bin/env python
# coding: utf-8

# # Projeto Covid-19
# 
# ## DIO-UNIMED

# In[ ]:


#importando bibliotecas

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


#importar dados para projeto

url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'


# In[ ]:


#ler dados

df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])
df.head(5)


# In[ ]:


#conferir tipos de dados em cada coluna

df.dtypes


# In[ ]:


#definindo variável para limpar nomes das colunas, retirando barra e letras maiúsculas

import re

def corrigindo_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()


# In[ ]:


#corrigindo todas as colunas do df

df.columns = [corrigindo_colunas(col) for col in df.columns]


# In[ ]:


df.head(5)


# In[ ]:


#selecionando Brasil para investigar

df.loc[df.countryregion == 'Brazil']


# In[ ]:


brasil = df.loc[
    (df.countryregion == 'Brazil') &
    (df.confirmed > 0)
]


# In[ ]:


brasil


# In[ ]:


#gráfico de casos confirmados

px.line(brasil, 'observationdate', 'confirmed', title='Casos confirmados no Brasil')


# In[ ]:


#criando coluna de novos casos por dia

brasil['novos casos'] = list(map(
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1],
    np.arange(brasil.shape[0])
))


# In[ ]:


#gráfico com a nova coluna

px.line(brasil, x='observationdate', y='novos casos', title='Novos casos por dia')


# In[ ]:


#gráfico com número de mortes

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes',
              mode='lines+markers', line={'color':'red'})

)

fig.update_layout(title='Mortes por Covid-19 no Brasil')

fig.show()


# In[ ]:


#calcular taxa de crescimento

#taxa de crescimento = (presente/passado)**(1/n) -1
def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    # Se data_inicio for None, define como a primeira data disponível no dataset
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
    
    # Define os valores de presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    
    # Define o número de pontos no tempo q vamos avaliar
    n = (data_fim - data_inicio).days
    
    # Calcula a taxa
    taxa = (presente/passado)**(1/n) - 1

    return taxa*100


# In[ ]:


#taxa de crescimento médio da Covid no Brasil

taxa_crescimento(brasil, 'confirmed')


# In[ ]:


def taxa_crescimento_diaria(data, variable, data_inicio=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    data_fim = data.observationdate.max()
    n = (data_fim - data_inicio).days
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        range(1,n+1)
    ))
    return np.array(taxas)*100


# In[ ]:


tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')


# In[ ]:


tx_dia


# In[ ]:


primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()
px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil',
       labels={'y':'Taxa de crescimento', 'x':'Data'})


# In[ ]:


#predições

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

confirmados = brasil.confirmed
confirmados.index = brasil.observationdate
confirmados


# In[ ]:


resultado = seasonal_decompose(confirmados)


# In[ ]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

ax1.plot(resultado.observed)
ax2.plot(resultado.trend)
ax3.plot(resultado.seasonal)
ax4.plot(confirmados.index, resultado.resid)
ax4.axhline(0, linestyle='dashed', c='black')

plt.show()


# In[ ]:


# ARIMA

get_ipython().system('pip install pmdarima')


# In[ ]:


from pmdarima.arima import auto_arima
modelo = auto_arima(confirmados)


# In[ ]:


fig = go.Figure(go.Scatter(
    x = confirmados.index, y=confirmados, name='Observados'
))

fig.add_trace(go.Scatter(
    x=confirmados.index, y=modelo.predict_in_sample(), name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forecast'
))

fig.update_layout(title='Previsão de casos confirmados no Brasil em 30 dias')
fig.show()


# In[ ]:


# modelo de crescimento

get_ipython().system('conda install -c conda-forge fbprophet -y')


# In[ ]:


from fbprophet import Prophet
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

#fbprophet exige que altere o nome das colunas

train.rename(columns={'observationdate':'ds', 'confirmed':'y'}, inplace = True)
test.rename(columns={'observationdate':'ds', 'confirmed':'y'}, inplace = True)

#definir modelo de crescimento

profeta = Prophet(growth='logistic', changepoints=['2020-03-21', '2020-03-20', '2020-04-25', '2020-05-03', '2020-05-10'])

pop = 211463256

#cenário em que todo mundo é contaminado
train['cap']=pop

#previsões
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast = profeta.predict(future_dates)


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
fig.add_trace(go.Scatter(x=test.index, y=test, name='Observados - Teste'))
fig.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
fig.update_layout(title='Predições de casos confirmados no Brasil')
fig.show()

