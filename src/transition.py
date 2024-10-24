""" Module that contains the Transition classes """
class Transition:
    """" Class that represents a transition between two roman numerals """
    def __init__(self, rn1, rn2, score:float=0):
        self.rn1_label, self.rn1_mode = rn1
        self.rn2_label, self.rn2_mode = rn2
        self.score = float(score)

    def __repr__(self):
        return f'{self.rn1_label} -> {self.rn2_label}: {self.score}'

    def __eq__(self, other):
        return self.rn1_label == other.rn1_label and self.rn2_label == other.rn2_label \
        and self.rn1_mode == other.rn1_mode and self.rn2_mode == other.rn2_mode

class Transitions:
    """ Class that represents a list of transitions """
    def __init__(self, *transitions:Transition):
        self.transitions = {}
        for tr in transitions:
            transition = Transition(*tr)
            self.transitions[transition.rn1_label+transition.rn1_mode+transition.rn2_label+transition.rn2_mode] = transition.score

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]

    def __repr__(self):
        return f'{self.transitions}'

    def get(self, idx):
        """ Returns the transition with the given index"""
        return self.transitions.get(idx)
