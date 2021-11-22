#!/usr/bin/python
# -*- coding: utf-8 -*-
# Path: asreviewcontrib\semantic_clustering\interactive.py

import pandas as pd

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input
import plotly.express as px


def run_app(filepath):
    """Function to be called to run the full Dash App"""

    # Load DataFrame with clusters
    df = pd.read_csv(filepath)

    # Read as STR for discrete colormap
    df['cluster_id'] = df['cluster_id'].astype(str)

    # Set 'cluster_id' to max if 'inclusion_label' == 1
    for row in df.itertuples():
        if row.included == 1:
            df.at[row.Index, 'cluster_id'] = 'included'

    # Show main figure
    fig = px.scatter(df,
                     x="x",
                     y="y",
                     color="cluster_id",
                     color_discrete_sequence=px.colors.qualitative.Light24)
    fig.update_layout(dragmode="pan")
    fig.update_layout(legend={'traceorder': 'normal'},
                      plot_bgcolor='rgba(0,0,0,0.35)',
                      height=400,)
    fig.update_layout(xaxis=dict(showticklabels=False, title=""),
                      yaxis=dict(showticklabels=False, ticks="", title=""))

    config = dict(
        {'scrollZoom': True,
            'displayModeBar': False,
            'displaylogo': False,
            'clear_on_unhover': True})

    # Initialize app and do lay-out
    app = dash.Dash()
    app.layout = html.Div([

        # banner div
        html.Div([
            html.H2("Visualizing Semantic Clusters"),
        ], className="banner"),

        # external css div
        html.Div([

            # Main semantic cluster graph
            html.Div([
                dcc.Graph(figure=fig, id="cluster-div", config=config,
                          style={'width': '100%',
                                 'height': '100%'
                                 },)
            ], className="six columns", style={'height': '80%'}),

            # Div for abstract window
            html.Div([
                html.H3("Test title", id="paper-title"),
                dcc.Textarea(
                    readOnly=True,
                    placeholder='Enter a value...',
                    value='This is a TextArea component',
                    style={'width': '98%', 'height': '389px'},
                    id="abstract-div"
                )
            ], className="six columns"),

        ], className="row", style={'height': '100%'}),
    ], style={'backgroundColor': 'rgba(0,0,0,0.1)',
              'position': 'fixed',
              'width': '100%',
              'height': '100%',
              'top': '0',
              'left': '0',
              'z-index': '10',
              'padding': '10px'
              })

    # Allow global css - use chriddyp's time-tested external css
    app.css.config.serve_locally = False
    app.css.append_css({
        "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
    })

    ############################
    # ### DEFINE CALLBACKS ### #
    ############################

    # Callback to refresh abstract window
    @app.callback(dash.dependencies.Output("abstract-div", "value"),
                  [Input('cluster-div', 'hoverData')])
    def update_abstract(hoverData):

        # Fetch df
        nonlocal df

        # Update graph with hoverData
        if hoverData is not None:
            # Get abstract based on x and y values
            x = hoverData['points'][0]['x']
            y = hoverData['points'][0]['y']
            return df[(df['x'] == x) & (df['y'] == y)
                      ]['abstract'].values[0]

        else:
            return df['abstract'].iloc[0]

    # Callback to refresh article title
    @app.callback(dash.dependencies.Output("paper-title", "children"),
                  [Input('cluster-div', 'hoverData')])
    def update_title(hoverData):

        # Fetch df
        nonlocal df

        # Update graph with hoverData
        if hoverData is not None:
            # Get title based on x and y values
            x = hoverData['points'][0]['x']
            y = hoverData['points'][0]['y']
            title = df[(df['x'] == x) & (df['y'] == y)
                       ]['title'].values[0]
            return title

        else:
            title = df['title'].iloc[0]

        return title

    # Run the application
    app.run_server(debug=False)
