"""Module to wrap the harmonic analyzer"""
import os
import base64
import verovio as vrv
import partitura as pt

from src.note_graph import NoteGraph
from src.rhythm_tree import RhythmTreeInteractive
from src.tonal_graph import TonalGraphInteractive
from src.mei_svg import MEISVGHandler

class HarmonicAnalyzer:
    """Class to wrap all the tools of the harmonic analyzer"""
    def __init__(self, vrv_toolkit: vrv.toolkit | None = None):
        self.vrv_toolkit = vrv_toolkit
        self.score = None   # partitura score
        self.note_graph = None
        self.rhythm_tree = None
        self.tonal_graph = None
        self.mei = None # MEI byte data
        self.svg_data = None    # SVG string data
        self.svg_doc = None    # MEISVGHandler document

    def analyze_file(self, filepath):
        """ Analyze a file """
        self.score = pt.load_score(filepath)
        self.mei = pt.save_mei(self.score)
        self.note_graph = NoteGraph(self.score)
        self.rhythm_tree = RhythmTreeInteractive.from_note_graph(self.note_graph)
        self.tonal_graph = TonalGraphInteractive(self.rhythm_tree)


    def analyze_svg(self, output_path = None):
        """ Create the SVG of the score """
        assert isinstance(self.vrv_toolkit, vrv.toolkit), "The vrv_toolkit must be initialized"
        self.vrv_toolkit.loadData(self.mei.decode())
        self.vrv_toolkit.redoLayout()
        self.svg_data = self.vrv_toolkit.renderToSVG()
        if output_path is not None:
            self.vrv_toolkit.renderToSVGFile(output_path)
        self.svg_doc = MEISVGHandler.parse_string(self.svg_data)

    def load_from_dash_input(self, filename, score_contents):
        """ Input:  dash upload: filename and contents"""
        _ , score_contents_string = score_contents.split(',')
        decoded = base64.b64decode(score_contents_string)
        temp_filepath = os.path.join('temp',filename)
        os.makedirs('temp', exist_ok=True)
        with open(temp_filepath, 'wb') as f:
            f.write(decoded)

        self.analyze_file(temp_filepath)
        self.analyze_svg()

        os.remove(temp_filepath)

    def is_empty(self):
        """Check if the analyzer is empty"""
        return self.score is None
