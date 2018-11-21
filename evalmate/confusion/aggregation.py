from . import confusion


class AggregatedConfusion(confusion.Confusion):
    """
    Class to aggregate multiple confusions.

    Attributes:
        instances (dict): Dictionary containing the aggregated confusions.
    """

    def __init__(self):
        self.instances = {}

    @property
    def correct(self):
        return sum([x.correct for x in self.instances.values()])

    @property
    def insertions(self):
        return sum([x.insertions for x in self.instances.values()])

    @property
    def deletions(self):
        return sum([x.deletions for x in self.instances.values()])

    @property
    def substitutions(self):
        return sum([x.substitutions for x in self.instances.values()])

    @property
    def substitutions_out(self):
        return sum([x.substitutions_out for x in self.instances.values()])
