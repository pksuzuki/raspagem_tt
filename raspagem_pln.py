# -*- coding: utf-8 -*-
"""
@author: PK
"""
from searchtweets import ResultStream,load_credentials,gen_request_parameters
import os
import json
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Módulo de PLN em pt-br
moduleName = 'leia'
leia = importlib.import_module(moduleName)

# Suas credenciais, fornecidas pelo Twitter
dict = {
        "SEARCHTWEETS_BEARER_TOKEN": os.path.join("cred","bearertoken.txt"),
        "SEARCHTWEETS_CONSUMER_KEY": os.path.join("cred","conskey.txt"),
        "SEARCHTWEETS_CONSUMER_SECRET": os.path.join("cred","conssec.txt"),
        "SEARCHTWEETS_ENDPOINT": os.path.join("cred","endpoint.txt")
        }

for key, value in dict.items():
    with open(value,'r') as creds:
        os.environ[key] = creds.read()

search = load_credentials()

query = gen_request_parameters("", # Sua query de busca
                                   results_per_call=100,
                                   tweet_fields="") # Os campos que deseja retornar

rs = ResultStream(request_parameters=query,
                    max_results=100,
                    max_pages=1,
                    **search)

# Pull de tweets pela API
tweets = list(rs.stream())

s = leia.SentimentIntensityAnalyzer()

tweets = list()

df = pd.DataFrame(tweets)

# Manipule seu Data Frame como convém

df = df.groupby('author_id')['text'].apply(''.join).reset_index()

df['scores'] = df['text'].apply(s.polarity_scores)

df['compound'] = df['scores'].apply(lambda x: x.get('compound'))

# Alternativa mais simples para representação

# conditions = [
#     (df['compound'] < 0 ),
#     (df['compound'] == 0),
#     (df['compound'] > 0)]
# choices = ['Desaprova', 'Neutro', 'Aprova']

# df['op'] = np.select(conditions, choices)

df_sem_inconclusivos = df[df['compound']!=0]
out = pd.cut(df_sem_inconclusivos['compound'], bins=np.linspace(-1,1,20), include_lowest=True)
dist = out.value_counts(normalize=True).reindex(out.cat.categories)*100
dist = dist.reset_index()
plt.figure(figsize=(8, 5))
sns.set_style("darkgrid")
g_dist=sns.barplot(x = 'index',y = 'compound', data=dist)
plt.ylabel('Porcentagem', size=13)
plt.xlabel('Pontuação de Sentimento', size=13)
plt.xticks(ha='right', rotation=39)
plt.show()

# Salve sua busca!
with open('tweets.json', 'w') as f:
    json.dump(tweets,f)