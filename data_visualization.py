from random import random as rand
import plotly
from plotly.graph_objs import *
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

cv_set = 'bank-crossvalidation_new.csv'

COLUMNS = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration",
           "campaign","pdays","previous","poutcome","y"]

df = pd.read_csv(cv_set)

df.info()
quit()

plt.figure()
df.plot.pie("y")



df_yes = df[str(df.y)=='yes']
df_no = df[str(df.y)=='no']

df.head(2)

fig = {
    'data': [
  		{
  			'x': df_yes.duration,
        	'y': df_yes.age,
        	'text': df_yes.y,
        	'mode': 'markers',
        	'name': 'yes'},
        {
        	'x': df_no.duration,
        	'y': df_no.age,
        	'text': df_no.y,
        	'mode': 'markers',
        	'name': 'no'}
    ],
    'layout': {
        'xaxis': {'title': 'Duration of call'},
        'yaxis': {'title': "Age of client"}
    }
}


plotly.offline.plot(fig)