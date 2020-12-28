#import dash
#import dash_html_components as html

#app = dash.Dash()

#app.layout = html.Div(children=[
#    html.H1('Hello Dash!')
#])

#if __name__ == '__main__':
#    app.run_server()

# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df1 = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')


@app.callback(
    Output('life-exp-vs-gdp', 'figure'),
    Input('year-slider', 'value')
)
def update_figure(selected_year):
 filterDf = df[df.year == selected_year]
 fig = px.scatter(filterDf, x="gdpPercap", y="lifeExp", size="pop", color="continent",
hover_name="country", log_x=True, size_max=60)
 fig.update_layout(transition_duration=500)
 return fig

@app.callback(
    Output('year-pop', 'figure'),
    [dash.dependencies.Input('life-exp-vs-gdp', 'hoverData')]
)
def updateCountry(hoverData):
    if not hoverData:
        country = ''
    else:
        country = hoverData['points'][0]['hovertext']

    filterDf = df[df.country == country]
    fig = px.bar(filterDf, x="year", y="pop", title='Year vs Pop: {}'.format(country))
    return fig

app.layout = html.Div(children=[
    html.H1(children='Hello Dash!!'),
    dcc.Tabs(id='tabs', value='Tab 2', children=[
        dcc.Tab(label='Tab 1', value='Tab 1', children=[
            dcc.Dropdown(options=[
                {'label': 'New York City', 'value': 'NYC'},
                {'label': 'Montréal', 'value': 'MTL'},
                {'label': 'San Francisco', 'value': 'SF'}
            ], value='MTL')
        ]),
        dcc.Tab(label='Tab 2', value='Tab 2', children=[
            html.Div(children=html.H1('Hi!'))
        ])
    ]),
    dcc.Graph(
        id='life-exp-vs-gdp',
        #figure=px.scatter(df, x="Amount", y="Fruit", color="City")
    ),
    dcc.Slider(
        id='year-slider', min=df['year'].min(), value=df['year'].min(),
        max=df['year'].max(), marks={str(year): str(year) for year in df['year'].unique()}, step=None
    ),
    dcc.Graph(
        id='year-pop',
        #figure=px.bar(df1, x="Fruit", y="Amount", color="City", barmode="group")
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)