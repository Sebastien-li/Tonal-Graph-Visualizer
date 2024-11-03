"""
Contains the classes for the music theory objects
"""

import numpy as np
from functools import cache

class Pitch:
    """
    Class that represents a pitch in the diatonic / chromatic space
    """
    note_names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    diatonic_dict = {note: i for i, note in enumerate(note_names)}
    chromatic_dict = {'C':0, 'D':2, 'E':4, 'F':5, 'G':7, 'A':9, 'B':11}

    def __init__(self, diatonic:int, chromatic:int):
        self.diatonic = diatonic%7
        self.chromatic = chromatic%12
        self.name_without_accidental = self.note_names[self.diatonic]
        accidental_number = (chromatic - self.chromatic_dict[self.name_without_accidental])%12
        if accidental_number <= 6:
            self.accidental = '#' * accidental_number
        else:
            self.accidental = '-' * (12-accidental_number)
        self.name = self.name_without_accidental + self.accidental

    @classmethod
    def from_step_alter(cls, step:str, alter:int):
        """ Creates a pitch from a note step and an alteration"""
        diatonic = cls.diatonic_dict[step]
        chromatic = (cls.chromatic_dict[step] + alter)%12
        return cls(diatonic, chromatic)

    @classmethod
    def from_name(cls, name:str):
        """ Creates a pitch from a note name"""
        name = ''.join([x for x in name if not x.isdigit()])
        name_without_accidental = name[0]
        accidental = name[1:].replace('b','-')
        assert name_without_accidental in cls.diatonic_dict, f'Invalid note name: {name}'
        assert all('#' == x for x in accidental) or all('-' == x for x in accidental)\
            , f'Invalid accidental: {accidental}'
        diatonic = cls.diatonic_dict[name_without_accidental]
        if accidental:
            accidental_number = len(accidental) * (1 if accidental[0] == '#' else -1)
        else:
            accidental_number = 0
        chromatic = (cls.chromatic_dict[name_without_accidental] +
                          accidental_number ) %12

        return cls(diatonic, chromatic)

    def __repr__(self):
        return f'{self.name}'

    def __eq__(self, other):
        return self.chromatic == other.chromatic and self.diatonic == other.diatonic

    def __add__(self, interval):
        diatonic = (self.diatonic + interval.diatonic)%7
        chromatic = (self.chromatic + interval.chromatic)%12
        return Pitch(diatonic, chromatic)

    def __hash__(self):
        return hash((self.diatonic, self.chromatic))


class Interval:
    """
    Class that represents an interval in the diatonic / chromatic space
    """

    def __init__(self, diatonic:int, chromatic:int):
        self.diatonic = diatonic
        self.chromatic = chromatic
        self.interval_number = self.diatonic + 1

    @classmethod
    def from_pitches(cls, pitch_start:Pitch, pitch_end:Pitch ):
        """ Creates an interval from two pitches"""
        diatonic = (pitch_end.diatonic - pitch_start.diatonic)%7
        chromatic = (pitch_end.chromatic - pitch_start.chromatic)%12
        return cls(diatonic, chromatic)

    def __repr__(self):
        return f'({self.diatonic}, {self.chromatic})'

    def __eq__(self, other):
        return self.diatonic == other.diatonic and self.chromatic == other.chromatic

    def __hash__(self):
        return hash((self.diatonic, self.chromatic))

class Quality:
    """
    Class that represents a chord quality and the score of each note in the chord
    """
    def __init__(self, label:str='NO', name:str='NO', score_dict:dict=None):
        self.label = label
        self.score_dict = score_dict
        self.name = name
        self.cardinality = len(score_dict if score_dict else {})

    def __repr__(self):
        return f'{self.label}'

    def label_with_inversion(self, inversion):
        """ Returns the label of the chord with the inversion"""
        if self.cardinality == 3:
            inversion_name = ['','6','64'][inversion]
            full_name = self.label + inversion_name
        elif self.cardinality  == 4:
            inversion_name = ['7','65','43','2'][inversion]
            full_name = self.label.replace('7',inversion_name)
        else:
            full_name = 'NO'
        return full_name

    def __eq__(self, other):
        return self.label == other.label

class Qualities:
    """
    Class that represents a collection of qualities
    """
    def __init__(self, *quality_list):
        self.quality_list = [Quality(*qu) for qu in quality_list]
        self.quality_dict = {quality.label:quality for quality in self.quality_list}
        self.idx_to_name = {i: quality.label for i, quality in enumerate(self.quality_list)}
        self.name_to_idx = {quality.label: i for i, quality in enumerate(self.quality_list)}
        self.pitch_beam = self.__compute_pitch_beat() #pitch -> [(root, quality) of the chord]
        self.chord_array = self.__compute_chord_array() #diatonic, chromatic, quality_idx -> chord
        self.len = len(self.quality_list)

    def __getitem__(self,label):
        if isinstance(label,int):
            return self.quality_list[label]
        if isinstance(label,str):
            return self.quality_dict[label]
        raise ValueError('Invalid label for qualities')

    def __iter__(self):
        return iter(self.quality_dict.items())

    def __repr__(self):
        return f'{list(self.quality_dict.keys())}'

    def __len__(self):
        return len(self.quality_list)

    def __compute_pitch_beat(self):
        pitch_beam = {}
        for quality_label, quality in self:
            for root_diatonic in range(7):
                for root_chromatic in range(12):
                    root = Pitch(root_diatonic, root_chromatic)
                    for interval_note_name, score in quality.score_dict.items():
                        interval = Interval.from_pitches(Pitch.from_name(interval_note_name),
                                                         Pitch.from_name('C'))
                        if root not in pitch_beam:
                            pitch_beam[root] = []
                        pitch_beam[root].append((root+interval,quality_label,score))
        return pitch_beam

    def __compute_chord_array(self):
        chord_array = np.zeros((7,12,len(self)), dtype=object)
        for root_diatonic in range(7):
            for root_chromatic in range(12):
                for quality_idx, (_, quality) in enumerate(self):
                    chord = {Pitch.from_name(note)+Interval(root_diatonic,root_chromatic): score
                             for note,score in quality.score_dict.items()}
                    chord_array[root_diatonic, root_chromatic, quality_idx] = chord
        return chord_array

class RomanNumeralFigure:
    """" Class that represents a roman numeral """
    def __init__(self, figure:str, diatonic_root:int, chromatic_root:int, quality:str, score:float):
        self.figure = figure
        self.diatonic_root = diatonic_root
        self.chromatic_root = chromatic_root
        self.quality = quality
        self.score = score
        self.label = self.get_label()
    def __repr__(self):
        return f'{self.label}'

    def get_label(self):
        ''' Converts the figure into a label '''
        label = self.figure
        if self.quality in ['m','o', 'm7', 'o7', 'ø7']:
            label = label.lower()
        if self.quality in ['M','m','It','Ger','Fr']:
            quality = ''
        elif self.quality == 'm7':
            quality = '7'
        else:
            quality = self.quality
        return label+quality

class Mode:
    """ Class that represents a mode (major or minor) """
    def __init__(self, name: str, roman_numeral_list: list):
        self.name = name
        self.roman_numeral_list = [RomanNumeralFigure(*rn) for rn in roman_numeral_list]

    def __iter__(self):
        return iter(self.roman_numeral_list)

    def __repr__(self):
        return f'{self.name}'

class Key:
    """ Class that represents a key """
    def __init__(self, tonic:Pitch, mode: str):
        self.tonic = tonic
        self.mode = mode
        self.neutral_tonic = Pitch(0,0) if mode == 'M' else Pitch(5,9)
        self.key_alteration = self.__compute_key_alteration()
    def __compute_key_alteration(self):
        return compute_key_alteration(self.tonic.diatonic,self.tonic.chromatic,
                                             self.neutral_tonic.diatonic, self.neutral_tonic.chromatic)
    def __repr__(self):
        if self.mode == 'M':
            return f'{self.tonic}'
        return f'{str(self.tonic).lower()}m'

@cache
def compute_key_alteration(diatonic, chromatic, diatonic_target, chromatic_target):
    """ Computes the alteration of a key """
    diatonic_minus, chromatic_minus = diatonic, chromatic
    diatonic_plus, chromatic_plus = diatonic, chromatic
    i = 0
    while True:
        if diatonic_minus == diatonic_target and chromatic_minus == chromatic_target:
            return -i
        if diatonic_plus == diatonic_target and chromatic_plus == chromatic_target:
            return i
        diatonic_minus, chromatic_minus = (diatonic_minus + 4)%7, (chromatic_minus + 7)%12
        diatonic_plus, chromatic_plus = (diatonic_plus - 4)%7, (chromatic_plus - 7)%12
        i += 1
