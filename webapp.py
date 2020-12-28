import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
from dash.dependencies import Input, Output
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')
df = pd.read_csv('dataset4.csv')

# PCA Analysis
n_components = 2
pca = PCA(n_components=n_components)
components = pca.fit_transform(dataset2)
total_var = pca.explained_variance_ratio_.sum() * 100
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

# Scree Plot
pca1 = PCA()
pc = pca1.fit_transform(dataset2)
sp = pd.DataFrame({'Variance Explained (%)': pca1.explained_variance_ratio_*100, 'Principle Component': ['PC1','PC2','PC3','PC4','PC5','PC6','PC7']})

# Biplot
features = ['total_enrollment', 'male_num', 'female_num', 'asian_num', 'white_num', 'hispanic_num', 'black_num']
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
bpfig=px.scatter(components, x=0, y=1, title='7 Projected Axes Biplot')
for i, feature in enumerate(features):
    bpfig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    bpfig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )

# MDS
similarities = pairwise_distances(dataset2, metric='euclidean')
mdsg = MDS(n_components=2, metric=True, dissimilarity='precomputed')
pos = mdsg.fit_transform(similarities)
dissimilarities = pairwise_distances(dataset2.corr(method='pearson'), metric='correlation')
sop = mdsg.fit_transform(dissimilarities)

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/correlation-matrix':
        return correlation_matrix_layout
    elif pathname == '/scatter-plot-matrix':
        return scatter_plot_layout
    elif pathname == '/parallel-coordinates-display':
        return parallel_coordinates_layout
    elif pathname == '/pca-scree-plot':
        return pca_plot_layout
    elif pathname == '/biplot':
        return biplot_layout
    elif pathname == '/MDS-display-data':
        return mds_data_layout
    elif pathname == '/MDS-display-attributes':
        return mds_attributes_layout
    elif pathname == '/basic-visualizations':
        return basic_visualizations_layout
    elif pathname == '/advanced-displays':
        return advanced_displays_layout
    elif pathname == '/interactive-dashboard':
        return interactive_dashboard_layout
    else:
        return index_page

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1('NYC Public School Demographics (2006-2018)', style={'text-align': 'center'}),
    dcc.Link('Basic Visualizations', href='/basic-visualizations'),
    html.Br(),
    dcc.Link('Advanced Displays', href='/advanced-displays'),
    html.Br(),
    dcc.Link('Interactive Dashboard', href='/interactive-dashboard'),
])

correlation_matrix_layout = html.Div([
    html.H1('Correlation Matrix Page', style={'text-align': 'center'}),
    dcc.Graph(
        figure=px.imshow(dataset2.corr(method='pearson'), color_continuous_scale='Picnic', range_color=[-1, 1],
                         labels=dict(color="corr"), title="7x7 Correlation Matrix", origin='lower')
    ),
    html.Br(),
    dcc.Link('Go back', href='/advanced-displays')
])

scatter_plot_layout = html.Div([
    html.H1('Scatter Plot Page', style={'text-align': 'center'}),
    dcc.Graph(
        figure=px.scatter_matrix(dataset2,
                                 dimensions=["total_enrollment", "female_num", "male_num", "asian_num", "hispanic_num"],
                                 title="5x5 Scatter Plot Matrix", labels={col: col.replace('_', ' ') for col in dataset2.columns})
    ),
    html.Br(),
    dcc.Link('Go back', href='/advanced-displays')
])

parallel_coordinates_layout = html.Div([
    html.H1('Parallel Coordinates Page', style={'text-align': 'center'}),
    dcc.Graph(
        figure=px.parallel_coordinates(dataset2,
                                       dimensions=['total_enrollment', 'male_num', 'female_num', 'asian_num', 'white_num', 'hispanic_num', 'black_num'],
                                       labels={"total_enrollment": "Enrollment", "male_num": "Male", "female_num": "Female",
                                               "asian_num": "Asian", "white_num": "White", "hispanic_num": "Hispanic", "black_num": "Black"},
                                       title="7 Axes Parallel Coordinate Display for Numerical Data")),
    dcc.Graph(
        figure=px.parallel_categories(dataset3,
                                      dimensions=['DBN', 'name', 'year'],
                                      labels={"DBN": "District Borough Number", "name": "Name", "year": "Year"},
                                      title="3 Axes Parallel Coordinate Display for Categorical Data")),
    html.Br(),
    dcc.Link('Go back', href='/advanced-displays')
])

pca_plot_layout = html.Div([
    html.H1('PCA & Scree Plot Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=px.scatter(components, x=0, y=1, labels=labels, title=f'Total Explained Variance: {total_var:.2f}%')),
    dcc.Graph(
        figure=px.bar(sp, x='Principle Component', y='Variance Explained (%)', title='Associated Scree Plot')),
    dcc.Link('Go back', href='/advanced-displays')
])

biplot_layout = html.Div([
    html.H1('Biplot Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=bpfig),
    dcc.Link('Go back', href='/advanced-displays')
])

mds_data_layout = html.Div([
    html.H1('MDS Data Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=px.scatter(pos, x=0, y=1, title='MDS Plot (Euclidian distance)')),
    dcc.Link('Go back', href='/advanced-displays')
])

mds_attributes_layout = html.Div([
    html.H1('MDS Attributes Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=px.scatter(sop, x=0, y=1, title='MDS Plot (1-|correlation| distance)')),
    dcc.Link('Go back', href='/advanced-displays')
])

basic_visualizations_layout = html.Div([
    html.H1('Basic Visualizations', style={'text-align': 'center'}),

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
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

advanced_displays_layout = html.Div([
    html.H1('Advanced Displays', style={'text-align': 'center'}),
    dcc.Link('Navigate to Correlation Matrix', href='/correlation-matrix'),
    html.Br(),
    dcc.Link('Navigate to Scatter Plot Matrix', href='/scatter-plot-matrix'),
    html.Br(),
    dcc.Link('Navigate to Parallel Coordinates Display', href='/parallel-coordinates-display'),
    html.Br(),
    dcc.Link('Navigate to PCA & Scree Plot', href='/pca-scree-plot'),
    html.Br(),
    dcc.Link('Navigate to Biplot', href='/biplot'),
    html.Br(),
    dcc.Link('Navigate to MDS Display of Data', href='/MDS-display-data'),
    html.Br(),
    dcc.Link('Navigate to MDS Display of Attributes', href='/MDS-display-attributes'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

interactive_dashboard_layout = html.Div([
    html.H1('Interactive Dashboard', style={'text-align': 'center'}),

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
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

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
    app.run_server(debug=True, port=8050)
