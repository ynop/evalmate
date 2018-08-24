import numpy as np


class Outcome:
    """
    An outcome represents the annotation/labels/transcriptions of a dataset/corpus for a given task.
    This can be either the ground truth/reference or the system output/hypothesis.

    Attributes:
        label_lists (dict): Dictionary containing all label-lists with the utterance/sample id as key.
    """

    def __init__(self, label_lists=None):
        self.label_lists = label_lists or {}

    def label_set(self):
        """ Return a label-set containing all labels. """
        ls = LabelSet()

        for label_list in self.label_lists.values():
            ls.labels.extend(label_list.labels)

        return ls

    def label_set_for_value(self, value):
        """
        Return a label-set containing all labels, where the value is `value`.

        Arguments:
            value (str): The value to filter.

        Returns:
            LabelSet: Label-set containing all labels with the given value.
        """
        ls = LabelSet()

        for label_list in self.label_lists.values():
            for label in label_list:
                if label.value == value:
                    ls.labels.append(label)

        return ls


class LabelSet:
    """
    Class to collect a bunch of labels.
    This is used to compute statistics over a defined set of labels.

    For example we want to compute the average length of all labels with the value 'music'.
    We can then collect all these in a label-set and perform the computation.
    """

    def __init__(self, labels=None):
        self.labels = labels or []

    @property
    def count(self):
        """ Return the number of labels. """
        return len(self.labels)

    @property
    def length_min(self):
        """ Return the length of the shortest label. """
        return np.min(self.label_lengths)

    @property
    def length_max(self):
        """ Return the length of the longest label. """
        return np.max(self.label_lengths)

    @property
    def length_mean(self):
        """ Return the mean length of all labels. """
        return np.mean(self.label_lengths)

    @property
    def length_median(self):
        """ Return the median of all label lengths. """
        return np.median(self.label_lengths)

    @property
    def length_variance(self):
        """ Return the variance of all label lengths. """
        return np.var(self.label_lengths)

    @property
    def label_lengths(self):
        """ Return a list containing all label lengths. """
        return [label.duration for label in self.labels]
