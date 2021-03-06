import numpy as np

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

    @property
    def precision_mean(self):
        """ Calculate mean precision of all instances. """
        per_instance = [conf.precision for conf in self.instances.values()]
        return np.mean(per_instance)

    @property
    def recall_mean(self):
        """ Calculate mean recall of all instances. """
        per_instance = [conf.recall for conf in self.instances.values()]
        return np.mean(per_instance)

    def get_confusion_with_instances(self, instances):
        """
        Return a new AggregatedConfusion with only the given instances.

        Args:
            instances (list): A list of strings containing the keys of
                              the instances to include in the new confusion.

        Returns:
            AggregatedConfusion: A confusion with only the given instances.
        """
        new_conf = AggregatedConfusion()

        for key in instances:
            if key in self.instances.keys():
                new_conf.instances[key] = self.instances[key]
            else:
                raise ValueError('Instance with key "{}" not found!'.format(key))

        return new_conf
