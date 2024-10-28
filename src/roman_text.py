""" Module for the RomanText class """
from src.music_theory_classes import Pitch

class RomanNumeral:
    """ Class for a roman numeral with the inversion and the key"""
    def __init__(self, degree, figure, quality, inversion, key_tonic, mode):
                                    #Example : viio/65 in D minor
        self.degree = degree                    # VII
        self.figure = figure                    # viio/7
        self.quality = quality                  # o/7
        self.inversion = inversion              # 1
        self.key_tonic = key_tonic              # D
        self.mode = mode                        # minor
        self.key_tonic_disp = str(key_tonic).upper() if mode == 'M' else str(key_tonic).lower() # d
        self.full_name = self.get_full_name()   # viio/65
        self.full_name_with_key = self.get_full_name_with_key()    # d: viio/65

    def get_full_name(self):
        """ Returns the full name of the roman numeral"""
        if self.quality.cardinality == 3:
            tg_inversion_name = ['','6','64'][self.inversion]
            full_name = self.figure + tg_inversion_name
        elif self.quality.cardinality  == 4:
            tg_inversion_name = ['7','65','43','2'][self.inversion]
            full_name = self.figure.replace('7',tg_inversion_name)
        else:
            full_name = 'NO'
        return full_name

    def get_full_name_with_key(self):
        """ Returns the full name of the roman numeral with the key"""
        return f'{self.key_tonic_disp}: {self.full_name}'

    def __repr__(self):
        return self.full_name_with_key

    def __eq__(self, other):
        key = ('figure', 'key_tonic', 'mode')
        return all(getattr(self, k) == getattr(other, k) for k in key)

    @classmethod
    def from_tonal_graph_node(cls, tonal_graph, node_idx:int):
        """Transforms a tonal graph node into a RomanNumeral object"""
        node = tonal_graph.nodes[node_idx]
        key_tonic = Pitch(int(node['tonic_diatonic']),int(node['tonic_chromatic']))
        tg_mode = [x for x in tonal_graph.mode_list if x.name == node['mode']][0]
        tg_rn_figure = [rn for rn in tg_mode if rn.label == node['label']][0]
        degree = tg_rn_figure.figure
        figure = tg_rn_figure.label
        quality = tonal_graph.qualities[tg_rn_figure.quality]
        inversion = node['inversion']
        return cls(degree, figure, quality, inversion, key_tonic, node['mode'])
