from pathlib import Path

from dash import Dash
from tqdm import tqdm



from src.app.layout import get_layout
from src.app.callbacks.choose_piece import choose_piece_callback

def main():
    """ Main function """
    app = Dash(__name__)
    app.layout = get_layout()
    choose_piece_callback(app)
    app.run(debug=True, port = 8052)

if __name__ == '__main__':
    main()