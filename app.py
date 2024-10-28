""" Main application file """
from dash import Dash
import verovio as vrv

from src.harmonic_analyzer import HarmonicAnalyzer
from src.app.layout import get_layout
from src.app.callbacks.display import display_callbacks

verovio_options = {
        'breaks': 'none',
        'header':  'none',
        'adjustPageHeight': True,
        'pageMarginBottom': 0,
        'pageMarginTop': 0,
        'scaleToPageSize': True,
    }

def main():
    """ Main function """
    vrv.enableLog(vrv.LOG_ERROR)
    vrv_toolkit = vrv.toolkit()
    vrv_toolkit.setOptions(verovio_options)
    harmonic_analyzer = HarmonicAnalyzer(vrv_toolkit = vrv_toolkit)
    app = Dash(__name__)
    app.layout = get_layout()
    display_callbacks(app, harmonic_analyzer)
    app.run(debug=True, port = 8052)

if __name__ == '__main__':
    main()