import collections

from . import confusion


class EventConfusion(confusion.Confusion):
    """
    Class to represent confusions of a specific instance (e.g. some class) based on label-to-label alignment.
    The insertions, deletions and so on represent the number of times a label was confused (or not).

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

    def substitutions_by_count(self):
        """
        Return a list of tuples (Substituted-value, Number-of-substitutions) ordered by number of substitutions
        descending.

        Returns:
            list: List of tuples.
        """
        subs = [(x, len(y)) for x, y in self.substitution_pairs.items()]
        return sorted(subs, key=lambda x: (-x[1], x[0]))
