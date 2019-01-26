import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='s654231', api_key='M5QbTaUkVZIgxWhG2og6')
dataset = pd.read_csv('LD2_QT201601~201712.csv')

te = 10000
print(dataset['q1s'][1:te])
DATEd = dataset['DATE'][1:te] + " " + dataset['TIME'][1:te] 

# Create and style traces
q1s = go.Scatter(
    x = list(DATEd),
    y = list(dataset['q1s'][1:te]),
    name = 'q1 solor radation',
    line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 2),
    opacity = 0.8
)
q2s = go.Scatter(
    x = list(DATEd),
    y = list(dataset['q2s'][1:te]),
    name = 'q2 solor radation',
    line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 2),
    opacity = 0.8
)
data = [q1s, q2s]

# Edit the layout
layout = dict(
    title='Time series with range slider and selectors',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),

    )
)


fig = dict(data=data , layout=layout)
py.iplot(fig)
