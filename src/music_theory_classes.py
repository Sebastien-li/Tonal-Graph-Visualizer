"""
Contains the classes for the music theory objects
"""

import numpy as np

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
