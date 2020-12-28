import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from dash.dependencies import Input, Output
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

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
    else:
        return index_page

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
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
])

correlation_matrix_layout = html.Div([
    html.H1('Correlation Matrix Page', style={'text-align': 'center'}),
    dcc.Graph(
        figure=px.imshow(dataset2.corr(method='pearson'), color_continuous_scale='Picnic', range_color=[-1, 1],
                         labels=dict(color="corr"), title="7x7 Correlation Matrix", origin='lower')
    ),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])

scatter_plot_layout = html.Div([
    html.H1('Scatter Plot Page', style={'text-align': 'center'}),
    dcc.Graph(
        figure=px.scatter_matrix(dataset2,
                              dimensions=["total_enrollment", "female_num", "male_num", "asian_num", "hispanic_num"],
                              title="5x5 Scatter Plot Matrix", labels={col: col.replace('_', ' ') for col in dataset2.columns})
    ),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
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
    dcc.Link('Go back to home', href='/'),
])

pca_plot_layout = html.Div([
    html.H1('PCA & Scree Plot Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=px.scatter(components, x=0, y=1, labels=labels, title=f'Total Explained Variance: {total_var:.2f}%')),
    dcc.Graph(
        figure=px.bar(sp, x='Principle Component', y='Variance Explained (%)', title='Associated Scree Plot')),
    dcc.Link('Go back to home', href='/'),
])

biplot_layout = html.Div([
    html.H1('Biplot Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=bpfig),
    dcc.Link('Go back to home', href='/'),
])

mds_data_layout = html.Div([
    html.H1('MDS Data Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=px.scatter(pos, x=0, y=1, title='MDS Plot (Euclidian distance)')),
    dcc.Link('Go back to home', href='/')
])

mds_attributes_layout = html.Div([
    html.H1('MDS Attributes Page', style={'text-align': 'center'}),
    html.Br(),
    dcc.Graph(
        figure=px.scatter(sop, x=0, y=1, title='MDS Plot (1-|correlation| distance)')),
    dcc.Link('Go back to home', href='/'),
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
