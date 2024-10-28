""" Module for the callback when clicking on a node of the rhythm tree """
from dash import Output, Input, State
from src.app.plotter import plot_time_graph

def click_rhythm_callback(app):
    """ Function for the callback when clicking on a node of the rhythm tree """

    @app.callback(
        Output('graph-time-graphs', 'figure'),
        Input('graph-score', 'clickData'),
        State('graph-time-graphs', 'figure'),
        prevent_initial_call=True
    )