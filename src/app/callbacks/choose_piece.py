""" Module for the callback when choosing a piece from the upload button """
import os
import base64

from dash import dcc, html, Input, Output, State
import verovio as vrv
import partitura as pt

from src.app.plotter import plot_score, plot_time_graph
from src.note_graph import NoteGraph
from src.rhythm_tree import RhythmTreeInteractive
from src.tonal_graph import TonalGraphInteractive

def choose_piece_callback(app):
    """ Function for the callback for choosing a piece from the upload button """

    @app.callback(
        Output('div-upload-output', 'children'),
        Output('graph-score', 'figure'),
        Output('graph-time-graphs', 'figure'),
        Input('upload-button', 'filename'),
        Input('upload-button', 'contents')
    )
    def choose_piece(filename, score_contents):
        """ Callback for choosing a piece from the upload button """
        if filename is None:
            return 'No file uploaded', {}, {}

        path_name = os.path.join('assets','scores',filename)
        if not os.path.exists(path_name):
            return 'File must be in assets/scores directory', {}, {}

        score = pt.load_score(path_name)
        note_graph = NoteGraph(score)
        rhythm_tree = RhythmTreeInteractive.from_note_graph(note_graph)
        tonal_graph = TonalGraphInteractive(rhythm_tree)
        # # _ , score_contents_string = score_contents.split(',')
        # # decoded = base64.b64decode(score_contents_string)
        # vrv_toolkit = vrv.toolkit()
        # vrv_toolkit.loadFile(path_name)
        # print('File loaded')
        # svg_url = os.path.join('assets','temp',filename.split('.')[0] + '.svg')
        # vrv_toolkit.renderToSVGFile(svg_url)

        score_figure = plot_score(os.path.join('assets','scores','page-scaled.svg'), note_graph)
        time_graph_figure = plot_time_graph(note_graph, rhythm_tree, tonal_graph)

        return filename, score_figure, time_graph_figure