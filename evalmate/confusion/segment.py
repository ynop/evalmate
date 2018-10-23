import collections
import itertools

from . import confusion


class SegmentConfusion(confusion.Confusion):
    """
    Class to represent confusions of a specific instance (e.g. some class) based on segments.
    The insertions, deletions and so on represent the time in seconds the instance was confused (or not).

    Argument:
        value (str): The value of the instance (e.g. the class "speech")

    Attributes:
        correct_segments (list): (List of Segment) Segments that are correct (ref == hyp).
        insertion_segments (list): (List of Segment) Segments that are insertions (ref = None, hyp = 'value').
        deletion_segments (list): (List of Segment) Segments that are deletions (ref = 'value', hyp = None)
        substitution_segments (Dict): Segments that are substitutions with other values
                                      (ref = 'value', hyp = 'other-value').
                                      Dict holding a list for every `other-value`.
        substitution_out_segments (Dict): Segments that are substitutions of other values
                                          (ref = 'other-value', hyp = 'value').
                                          Dict holding a list for every `other-value`.
    """

    def __init__(self, value):
        self.value = value

        self.correct_segments = []
        self.insertion_segments = []
        self.deletion_segments = []
        self.substitution_segments = collections.defaultdict(list)
        self.substitution_out_segments = collections.defaultdict(list)

    @property
    def correct(self):
        return sum([x.duration for x in self.correct_segments])

    @property
    def insertions(self):
        return sum([x.duration for x in self.insertion_segments])

    @property
    def deletions(self):
        return sum([x.duration for x in self.deletion_segments])

    @property
    def substitutions(self):
        return sum([x.duration for x in itertools.chain(*self.substitution_segments.values())])

    @property
    def substitutions_out(self):
        return sum([x.duration for x in itertools.chain(*self.substitution_out_segments.values())])
