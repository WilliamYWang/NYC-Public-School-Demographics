import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('dataset4.csv')

app.layout = html.Div([
    html.H1('Lab 5', style={'text-align': 'center'}),

    html.Div([
        dcc.Graph(
            id='pie-chart',
        )], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='bar',
        )], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='scatter-graph',
        )], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='scatter-matrix',
        )], style={'width': '49%', 'display': 'inline-block'}),

    html.Div(dcc.Slider(
        id='year-slider', min=df['year'].min(), value=df['year'].min(),
        max=df['year'].max(), marks={str(year): str(year) for year in df['year'].unique()}, step=None
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])

def interactivity(selected_year, selected_borough, selected_schools, selected_race):
    filterDf = df[df.year == selected_year]

    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if input_id == 'pie-chart':
        borough = selected_borough['points'][0]['label']
        filterDf = filterDf[filterDf.borough == borough]
    elif input_id == 'scatter-graph':
        lst = selected_schools['points']
        schools = [d['hovertext'] for d in lst]
        filterDf = filterDf[filterDf['name'].isin(schools)]
    elif input_id == 'scatter-matrix':
        lst = selected_race['points']
        schools = [d['hovertext'] for d in lst]
        filterDf = filterDf[filterDf['name'].isin(schools)]

    return filterDf

@app.callback(
    Output('scatter-matrix', 'figure'),
    [Input('year-slider', 'value'), Input('pie-chart', 'clickData'), Input('scatter-graph', 'selectedData')])
def update_scattermatrix(selected_year, selected_borough, selected_schools):
    time.sleep(1)
    filterDf = interactivity(selected_year, selected_borough, selected_schools, '')
    filterDf.columns = ['DBN', 'School', 'year', 'total_enrollment', 'Female', 'female_per', 'Male', 'male_per',
                        'Asian', 'asian_per', 'Black', 'black_per', 'White', 'white_per', 'Hispanic', 'hispanic_per', 'borough']
    figure = px.scatter_matrix(filterDf,
                               dimensions=['Asian', 'Hispanic', 'White', 'Black'],
                               title='Race', hover_name='School')
    return figure

@app.callback(
    Output('bar', 'figure'),
    [Input('year-slider', 'value'), Input('pie-chart', 'clickData'), Input('scatter-graph', 'selectedData'),
     Input('scatter-matrix', 'selectedData')])
def update_bar(selected_year, selected_borough, selected_schools, selected_race):
    filterDf = interactivity(selected_year, selected_borough, selected_schools, selected_race)
    filterDf = filterDf[['asian_num', 'black_num', 'white_num', 'hispanic_num']]
    figure = px.bar(filterDf, x=[filterDf['white_num'].sum(), filterDf['asian_num'].sum(), filterDf['black_num'].sum(),
                    filterDf['hispanic_num'].sum()], y=['White Students', 'Asian Students', 'Black Students', 'Hispanic Students'],
                    title='Student Enrollment by Race')
    return figure

@app.callback(
    Output('scatter-graph', 'figure'),
    [Input('year-slider', 'value'), Input('pie-chart', 'clickData'), Input('scatter-matrix', 'selectedData')])
def update_scatter(selected_year, selected_borough, selected_race):
    filterDf = interactivity(selected_year, selected_borough, '', selected_race)
    filterDf.columns = ['DBN', 'School', 'year', 'total_enrollment', 'Female', 'female_per', 'Male', 'male_per',
                        'Asian', 'asian_per', 'Black', 'black_per', 'White', 'white_per', 'Hispanic', 'hispanic_per', 'borough']
    figure = px.scatter(filterDf, x='Male', y='Female', hover_name='School', title='Gender')
    return figure

@app.callback(
    Output('pie-chart', 'figure'),
    [Input('year-slider', 'value'), Input('scatter-graph', 'selectedData'), Input('scatter-matrix', 'selectedData')])
def update_pie(selected_year, selected_schools, selected_race):
    filterDf = interactivity(selected_year, '', selected_schools, selected_race)
    figure = px.pie(filterDf, values='total_enrollment', names='borough', title='Student Enrollment by Borough')
    return figure

if __name__ == '__main__':
    app.run_server(debug=True, port=8053)