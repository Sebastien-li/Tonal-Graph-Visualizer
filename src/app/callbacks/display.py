""" Module for the callback when choosing a piece from the upload button """
from time import time

from dash import Input, Output, State, no_update

from src.app.plotter import plot_score, plot_time_graph
from src.harmonic_analyzer import HarmonicAnalyzer
from src.app.layout import visible_style

sticky_style = {'display': 'block', 'position':'sticky',
                    'top': '-100px', 'z-index': '1',
                    'background-color': 'white'}

def display_callbacks(app, harmonic_analyzer: HarmonicAnalyzer):
    """ Function for the callback for choosing a piece from the upload button """

    @app.callback(
        Output('div-upload-output', 'children'),
        Output('graph-score', 'figure'),
        Output('graph-time-graphs', 'figure'),
        Output('div-score', 'style'),
        Output('div-time-graphs', 'style'),
        Output('trace-index', 'data'),
        Input('upload-button', 'filename'),
        Input('upload-button', 'contents'),
        State('check-sticky-score', 'value'),
    )
    def choose_piece(filename, score_contents, sticky):
        """ Callback for choosing a piece from the upload button """
        if filename is None:
            return no_update, no_update, no_update, no_update, no_update, no_update
        print(f'File chosen: {filename}')
        t0 = time()
        harmonic_analyzer.load_from_dash_input(filename, score_contents)
        t1 = time()
        print(f"Time to load and analyze the file: {t1-t0:.2f} s")
        score_figure = plot_score(harmonic_analyzer)
        t2 = time()
        print(f"Time to plot the score: {t2-t1:.2f} s")
        time_graph_figure, trace_index = plot_time_graph(harmonic_analyzer)
        t3 = time()
        print(f"Time to plot the time graph: {t3-t2:.2f} s\n")
        score_style = sticky_style if 'Sticky score' in sticky else visible_style
        return filename, score_figure, time_graph_figure, score_style, visible_style, trace_index

    @app.callback(
        Output('div-score', 'style', allow_duplicate=True),
        Input('check-sticky-score', 'value'),
        prevent_initial_call=True
    )
    def sticky_score(sticky):
        return sticky_style if 'Sticky score' in sticky else visible_style