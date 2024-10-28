""" This module contains the layout of the app. """
from dash import dcc, html

invisible_style = {'display':'none'}
visible_style = {'display':'block'}

def get_layout():
    """ Return the layout of the app."""

    layout = [
        dcc.Loading(id = 'loading', type='circle', children = [

            html.H1(id = 'h1-title', children = 'Tonal Graph Visualization'),

            html.Div(id = 'div-upload', children = [
                dcc.Upload(id = 'upload-button', children = html.Button('Upload File')),
                html.Div(id = 'div-upload-output', children = 'No file uploaded')
            ]),

            html.Div(id = 'div-score', children = [
                dcc.Graph(id = 'graph-score', figure = {})
            ], style = invisible_style),

            html.Div(id = 'div-time-graphs', children = [
                dcc.Graph(id = 'graph-time-graphs', figure = {}),
            ], style = invisible_style),

        ], overlay_style={"visibility":"visible", "opacity": .5, "backgroundColor": "white"}),



    ]
    return layout
