""" Main file for the project. """
from time import time
from pathlib import Path

import partitura as pt
import verovio as vrv

from src.note_graph import NoteGraph
from src.rhythm_tree import RhythmTreeInteractive
from src.tonal_graph import TonalGraphInteractive
from src.utils import get_multilogger


def main():
    """ Main function"""
    logger = get_multilogger()
    file_path = Path(r'assets\scores\Beethoven\Theme and Variations WoO78\score.mxl')
    mei_path = file_path.parent / 'score.mei'

    t0 = time()

    score = pt.load_score(str(file_path))
    pt.save_mei(score, mei_path)
    logger.info('MEI file created in %s seconds', time()-t0)

    t1 = time()
    note_graph = NoteGraph(score)
    logger.info('Note graph created in %s seconds', time()-t1)

    t2 = time()
    rhythm_tree = RhythmTreeInteractive.from_note_graph(note_graph)
    logger.info('Created and analyzed rhythm tree in %s seconds', time()-t2)

    t3 = time()
    tonal_graph = TonalGraphInteractive(rhythm_tree)
    logger.info('TonalGraph created in %s seconds', time()-t3)


    vrv_toolkit = vrv.toolkit()
    vrv_toolkit.loadFile(str(mei_path))
    print(vrv_toolkit)
if __name__ == '__main__':
    main()