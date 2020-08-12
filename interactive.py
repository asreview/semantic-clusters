# Imports

# System / os stuff
import os

# Data
import numpy as np 
import pandas as pd 

# Dash-y
import dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Output, Input

# Plotly
import plotly.graph_objs as go
import plotly.express as px

def run_app():
    """Function to be called to run the full Dash App"""
    
    # Load DataFrame with clusters
    df_path = os.path.join("data","dataframes","kmeans_df.csv")
    df = pd.read_csv(df_path)

    # Read as STR for discrete colormap
    df['cluster_id'] = df['cluster_id'].astype(str)

    # Show main figure
    #fig = px.scatter(df, x="x", y="y", color="cluster_id", color_discrete_map=px.colors.sequential.Viridis)
    fig = px.scatter(df, x="x", y="y", color="cluster_id", color_discrete_sequence=px.colors.qualitative.T10)
    fig.update_layout(dragmode="pan")
    fig.update_layout(xaxis=dict(showticklabels=False, title=""), 
                      yaxis=dict(showticklabels=False, ticks="", title=""))
    config = dict({'scrollZoom': True, 'displayModeBar':False, 'displaylogo':False})
    # fig.show(config=config)

    # Initialize app and do lay-out
    app = dash.Dash()
    app.layout = html.Div([

        # banner div
        html.Div([
            html.H2("CORD-19: Visualizing Semantic Clusters"),
            #html.Img(src="/assets/stock-icon.png")
        ], className="banner"),

        # external css div
        html.Div([

            # Main semantic cluster graph
            html.Div([
                dcc.Graph(figure=fig,id="cluster-div",config=config)
            ], className = "six columns"),

            # Div for abstract window
            html.Div([ 
                html.H3("Test Title", id="paper-title"),
                dcc.Textarea(
                    readOnly=True,
                    placeholder='Enter a value...',
                    value='This is a TextArea component',
                    style={'width': '100%','height':'300px'},
                    id="abstract-div"
                )  
            ], className = "six columns"),

        ], className="row"),
    ])

    # Allow global css - use chriddyp's time-tested external css
    app.css.config.serve_locally = False
    app.css.append_css({
        "external_url":"https://codepen.io/chriddyp/pen/bWLwgP.css"
    })

    ############################
    ##### DEFINE CALLBACKS #####
    ############################

    # Callback to refresh Abstract window
    @app.callback(dash.dependencies.Output("abstract-div", "value"),
                [Input('cluster-div', 'hoverData')])
    def update_abstract(hoverData):

        # Fetch df
        nonlocal df

        # Update graph with hoverData
        if hoverData != None:
            hover_dict = hoverData['points'][0]
            abstract_idx = hover_dict['pointIndex']

            # Set variable for abstract window update
            #cord_uid = df['cord_uid'].iloc[abstract_idx]
            abstract = df['Abstract'].iloc[abstract_idx]

            # Set hoverData to None again to prevent issues with graph update
            hoverData = None
        else:
            #cord_uid = df['cord_uid'].iloc[0]
            abstract = df['Abstract'].iloc[0]

        #return cord_uid
        return abstract

    # Callback to refresh Abstract window
    @app.callback(dash.dependencies.Output("paper-title", "children"),
                [Input('cluster-div', 'hoverData')])
    def update_title(hoverData):

        # Fetch df
        nonlocal df

        # Update graph with hoverData
        if hoverData != None:
            hover_dict = hoverData['points'][0]
            title_idx = hover_dict['pointIndex']

            # Set variable for paper title update
            title = df['Title'].iloc[title_idx]

            # Set hoverData to None again to prevent issues with graph update
            hoverData = None
        else:
            #cord_uid = df['cord_uid'].iloc[0]
            title = df['Title'].iloc[0]

        return title

    # Run the application
    app.run_server(debug=True)

if __name__ == "__main__":
    run_app()


"""Goes into update_abstract function"""
# # Make temp chart - we just want to change title for now
# temp_chart = go.Scatter(
#     x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#     y = [1,4,9,16,25,36,49,64,81,100,121,144,169,196,225],
#     name="abstract-window",
# )

# temp_chart = dcc
# temp_chart = [temp_chart]

# # Change layout so we can change title
# abstract_layout = dict(
#     title=f"{cord_uid}"
# )

# abstract_fig = dict(data=temp_chart, layout=abstract_layout)