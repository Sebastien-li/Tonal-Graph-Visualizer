"""
Contains the objects that define the music theory of the model.
"""
from src.music_theory_classes import Qualities, Mode, Pitch
from src.transition import Transitions

qualities = Qualities(('M','major triad', {'C':0.4, 'E':0.3, 'G':0.3}),
                      ('m','minor triad', {'C':0.4, 'E-':0.3, 'G':0.3}),
                      ('o','diminished triad', {'C':0.38, 'E-':0.3, 'G-':0.32}),
                      ('+','augmented triad', {'C':0.37, 'E':0.3, 'G#':0.33}),
                      ('maj7','major seventh chord', {'C':0.35, 'E':0.2, 'G':0.2, 'B':0.25}),
                      ('m7','minor seventh chord', {'C':0.35, 'E-':0.2, 'G':0.2, 'B-':0.25}),
                      ('7','dominant seventh chord', {'C':0.35, 'E':0.2, 'G':0.2, 'B-':0.25}),
                      ('o7','diminished seventh chord', {'C':0.35, 'E-':0.1, 'G-':0.3, 'B--':0.25}),
                      ('ø7','half-diminished seventh chord', {'C':0.35, 'E-':0.1, 'G-':0.3, 'B-':0.25}),
                      ('It','Italian augmented sixth chord', {'C':0.39,'E--':0.39,'G-':0.22}),
                      ('Fr','French augmented sixth chord', {'C':0.11,'E':0.39,'G-':0.39,'B-':0.11}),
                      ('Ger','German augmented sixth chord', {'C':0.39,'E--':0.39,'G-':0.11,'B--':0.11}),
                      ('','other', {'C':0,'C#':0,'D':0,'D#':0,'E':0,'F':0,'F#':0,'G':0,'G#':0,'A':0,'A#':0,'B':0}))

mode_list = [
    Mode('M',
        [('I',0,0,'M',1),
        ('I',0,0,'maj7',0.5), # ?
        ('II',1,2,'m',0.99),
        ('II',1,2,'m7',0.99),
        ('III',2,4,'m',0.8),
        ('III',2,4,'m7',0.5), # ?
        ('IV',3,5,'m',0.9),
        ('IV',3,5,'M',0.99),
        ('IV',3,5,'maj7',0.8),
        ('V',4,7,'M',0.99),
        ('V',4,7,'7',0.99),
        ('VI',5,9,'m',0.99),
        ('VI',5,9,'m7',0.8),
        ('VII',6,11,'o',0.99),
        ('VII',6,11,'ø7',0.99),])
    ,
    Mode('m',
        [('I',0,0,'m',1),
        ('I',0,0,'m7',0.8),
        ('N',1,1,'M',0.95), #Napolitan
        ('II',1,2,'o',0.99),
        ('II',1,2,'ø7',0.99),
        ('III',2,3,'+',0.9),
        ('III',2,3,'M',0.6),
        ('III',2,3,'maj7',0.4), # ?
        ('IV',3,5,'m',0.99),
        ('IV',3,5,'m7',0.8),
        ('V',4,7,'M',0.99),
        ('V',4,7,'7',0.99),
        ('V',4,7,'m',0.8),
        ('VI',5,8,'M',0.99),
        ('VI',5,8,'maj7',0.6), # ?
        ('VII',6,11,'o',0.99),
        ('VII',6,11,'o7',0.99),
        ('VII',6,10,'M',0.8),
        ('It',3,6,'It',0.9),
        ('Ger',3,6,'Ger',0.9),
        ('Fr',1,2,'Fr',0.9)])
]
transitions = Transitions(
    (('V7','M'),('I','M'), 1),
    (('V','M'),('I','M'), 0.6),
    (('V7','m'),('i','m'), 1),
    (('V','m'),('i','m'), 0.6),
    (('viiø7','M'),('I','M'), 0.5),
    (('viio','M'),('I','M'), 0.6),
    (('viio7','m'),('i','m'), 0.5),
    (('viio','m'),('i','m'), 0.6)
)
