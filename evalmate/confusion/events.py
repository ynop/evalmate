import collections

from . import base


class EventConfusion(base.Confusion):
    """
    Class to represent confusions of a specific instance (e.g. some class) based on events/occurrences.

    Argument:
        value (str): The value of the instance (e.g. the class "speech")

    Attributes:
        correct_pairs (list): (List of LabelPair) Correct matches.
        insertion_pairs (list): (List of LabelPair) Insertions (ref = None, hyp = value)
        deletion_pairs (list): (List of LabelPair) Deletions (ref = value, hyp = None)
        substitution_pairs (Dict): Substitutions with other values (ref = value, hyp = other-value).
                                   Dict holding a list for every `other-value`.
        substitution_out_pairs (Dict): Substitutions from other values (ref = other-value, hyp = value)
                                       Dict holding a list for every `other-value`.
    """

    def __init__(self, value):
        self.value = value

        self.correct_pairs = []
        self.insertion_pairs = []
        self.deletion_pairs = []
        self.substitution_pairs = collections.defaultdict(list)
        self.substitution_out_pairs = collections.defaultdict(list)

    @property
    def correct(self):
        return len(self.correct_pairs)

    @property
    def insertions(self):
        return len(self.insertion_pairs)

    @property
    def deletions(self):
        return len(self.deletion_pairs)

    @property
    def substitutions(self):
        return sum([len(x) for x in self.substitution_pairs.values()])

    @property
    def substitutions_out(self):
        return sum([len(x) for x in self.substitution_out_pairs.values()])
