import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dataset1 = pd.read_csv('dataset1.csv')

@app.callback(
    Output('graph1', 'figure'),
    Input('dropdown1', 'value'))
def update_figure1(selected_variable):
    if selected_variable == 'year':
        s = dataset1['year'].value_counts()  # returns series containing counts of unique values
        tft = pd.DataFrame({'Year': s.index, 'Frequency': s.values})  # converts series to pandas dataframe with two columns (year, frequency)
        figure = px.bar(tft, x='Year', y='Frequency', title='Bar chart of ' + selected_variable)
    else:
        figure = px.histogram(dataset1, x=selected_variable, nbins=15, title='Histogram of ' + selected_variable)  # selected_variable is the value
    return figure

@app.callback(
    Output('graph2', 'figure'),
    Input('dropdown2', 'value'))
def update_figure2(selected_variable):
    if selected_variable == 'year':
        dataset1['year_group'] = pd.cut(dataset1['year'], bins=[2005, 2007, 2009, 2011, 2014, 2016, 2018],
        labels=['2006-2007', '2008-2009', '2010-2011', '2012-2014', '2015-2016', '2017-2018']) # bins values into intervals
        s = dataset1['year_group'].value_counts()
        tft = pd.DataFrame({'Years': s.index, 'Count': s.values})
        figure = px.pie(tft, names='Years', values='Count', title="Years")
    elif selected_variable == 'total_enrollment':
        dataset1['enrollment_group'] = pd.cut(dataset1['total_enrollment'], bins=[0, 499, 999, 1499, 1999, 4999],
        labels=['0-499', '500-999', '1000-1499', '1500-1999', '2000+'], include_lowest=True)
        s = dataset1['enrollment_group'].value_counts()
        tft = pd.DataFrame({'Enrollment Number': s.index, 'Frequency': s.values})
        figure = px.pie(tft, names='Enrollment Number', values='Frequency', title="Enrollment Count")
    elif selected_variable == 'female_num':
        dataset1['female_group'] = pd.cut(dataset1['female_num'], bins=[0, 199, 399, 599, 799, 999, 2399],
        labels=['0-199', '200-399', '400-599', '600-799', '800-999', '1000+'], include_lowest=True)
        s = dataset1['female_group'].value_counts()
        tft = pd.DataFrame({'Female Students': s.index, 'Frequency': s.values})
        figure = px.pie(tft, names='Female Students', values='Frequency', title="Frequency of Female Students")
    elif selected_variable == 'male_num':
        dataset1['male_group'] = pd.cut(dataset1['male_num'], bins=[0, 199, 399, 599, 799, 999, 2599],
        labels=['0-199', '200-399', '400-599', '600-799', '800-999', '1000+'], include_lowest=True)
        s = dataset1['male_group'].value_counts()
        tft = pd.DataFrame({'Male Students': s.index, 'Frequency': s.values})
        figure = px.pie(tft, names='Male Students', values='Frequency', title="Frequency of Male Students")
    elif selected_variable == 'asian_num':
        dataset1['asian_group'] = pd.cut(dataset1['asian_num'], bins=[0, 199, 399, 599, 799, 1999],
        labels=['0-199', '200-399', '400-599', '600-799', '800+'], include_lowest=True)
        s = dataset1['asian_group'].value_counts()
        tft = pd.DataFrame({'Asian Students': s.index, 'Frequency': s.values})
        figure = px.pie(tft, names='Asian Students', values='Frequency', title="Frequency of Asian Students")
    elif selected_variable == 'black_num':
        dataset1['black_group'] = pd.cut(dataset1['black_num'], bins=[0, 199, 399, 599, 799, 2599],
        labels=['0-199', '200-399', '400-599', '600-799', '800+'], include_lowest=True)
        s = dataset1['black_group'].value_counts()
        tft = pd.DataFrame({'Black Students': s.index, 'Frequency': s.values})
        figure = px.pie(tft, names='Black Students', values='Frequency', title="Frequency of Black Students")
    elif selected_variable == 'hispanic_num':
        dataset1['hispanic_group'] = pd.cut(dataset1['hispanic_num'], bins=[0, 199, 399, 599, 799, 2599],
        labels=['0-199', '200-399', '400-599', '600-799', '800+'], include_lowest=True)
        s = dataset1['hispanic_group'].value_counts()
        tft = pd.DataFrame({'Hispanic Students': s.index, 'Frequency': s.values})
        figure = px.pie(tft, names='Hispanic Students', values='Frequency', title="Frequency of Hispanic Students")
    elif selected_variable == 'white_num':
        dataset1['white_group'] = pd.cut(dataset1['white_num'], bins=[0, 199, 399, 599, 799, 3499],
        labels=['0-199', '200-399', '400-599', '600-799', '800+'], include_lowest=True)
        s = dataset1['white_group'].value_counts()
        tft = pd.DataFrame({'White Students': s.index, 'Frequency': s.values})
        figure = px.pie(tft, names='White Students', values='Frequency', title="Frequency of White Students")
    else:
        figure = px.pie(dataset1, names=selected_variable)

    return figure

@app.callback(
    Output('graph3', 'figure'),
    [Input('x-axis', 'value'), Input('y-axis', 'value')])
def update_figure3(xaxis, yaxis):
    figure = px.scatter(dataset1, x=dataset1[xaxis], y=dataset1[yaxis], hover_name='name', title='Scatter plot of ' + xaxis + ' vs ' + yaxis)
    figure.update_xaxes(title=xaxis)
    figure.update_yaxes(title=yaxis)
    return figure

app.layout = html.Div([
    html.H1('Lab 2', style={'text-align': 'center'}),

    dcc.Dropdown(id='dropdown1', options=[
        {'label': 'Year', 'value': 'year'},
        {'label': 'Total Enrollment', 'value': 'total_enrollment'},
        {'label': 'Female', 'value': 'female_num'},
        {'label': 'Male', 'value': 'male_num'},
        {'label': 'Asian', 'value': 'asian_num'},
        {'label': 'Black', 'value': 'black_num'},
        {'label': 'Hispanic', 'value': 'hispanic_num'},
        {'label': 'White', 'value': 'white_num'}
    ], value='year'),

    dcc.Graph(
        id='graph1'
    ),

    html.Hr(),

    dcc.Dropdown(id='dropdown2', options=[
        {'label': 'Year', 'value': 'year'},
        {'label': 'Total Enrollment', 'value': 'total_enrollment'},
        {'label': 'Female', 'value': 'female_num'},
        {'label': 'Male', 'value': 'male_num'},
        {'label': 'Asian', 'value': 'asian_num'},
        {'label': 'Black', 'value': 'black_num'},
        {'label': 'Hispanic', 'value': 'hispanic_num'},
        {'label': 'White', 'value': 'white_num'}
    ], value='year'),

    dcc.Graph(
        id='graph2'
    ),

    html.Hr(),

    html.Div([
        dcc.Dropdown(
            id='x-axis', options=[
            {'label': 'Year', 'value': 'year'},
            {'label': 'Total Enrollment', 'value': 'total_enrollment'},
            {'label': 'Female', 'value': 'female_num'},
            {'label': 'Male', 'value': 'male_num'},
            {'label': 'Asian', 'value': 'asian_num'},
            {'label': 'Black', 'value': 'black_num'},
            {'label': 'Hispanic', 'value': 'hispanic_num'},
            {'label': 'White', 'value': 'white_num'}
        ], value='female_num'),
        html.Div('x-axis', id='x-label')
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Dropdown(
            id='y-axis', options=[
            {'label': 'Year', 'value': 'year'},
            {'label': 'Total Enrollment', 'value': 'total_enrollment'},
            {'label': 'Female', 'value': 'female_num'},
            {'label': 'Male', 'value': 'male_num'},
            {'label': 'Asian', 'value': 'asian_num'},
            {'label': 'Black', 'value': 'black_num'},
            {'label': 'Hispanic', 'value': 'hispanic_num'},
            {'label': 'White', 'value': 'white_num'}
        ], value='male_num'),
        html.Div('y-axis', id='y-label')
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(
        id='graph3'
    ),

])

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)