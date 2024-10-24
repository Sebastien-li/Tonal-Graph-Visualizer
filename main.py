""" Main file for the project. """
from time import time
from pathlib import Path

import partitura as pt

from src.note_graph import NoteGraph
from src.rhythm_tree import RhythmTreeInteractive
from src.tonal_graph import TonalGraphInteractive
from src.utils import get_multilogger


def main(force_mei_export=True):
    """ Main function"""
    logger = get_multilogger()
    file_path = Path(r'assets\scores\Beethoven\Theme and Variations WoO78\score.mxl')
    mei_path = file_path.parent / 'score.mei'

    t0 = time()
    score = pt.load_score(str(file_path))
    logger.info('Score loaded in %s seconds', time()-t0)

    if not (mei_path).exists() or force_mei_export:
        t01 = time()
        pt.save_mei(score, mei_path)
        logger.info('MEI file created in %s seconds', time()-t01)

    t1 = time()
    note_graph = NoteGraph(score)
    logger.info('Note graph created in %s seconds', time()-t1)

    t2 = time()
    rhythm_tree = RhythmTreeInteractive.from_note_graph(note_graph)
    logger.info('Created and analyzed rhythm tree in %s seconds', time()-t2)

    t3 = time()
    tonal_graph = TonalGraphInteractive(rhythm_tree)
    logger.info('TonalGraph created in %s seconds', time()-t3)


if __name__ == '__main__':
    main()