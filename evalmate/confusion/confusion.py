import abc


class Confusion(abc.ABC):
    """
    Base class that provides methods for computing common metrics.
    """

    @property
    @abc.abstractmethod
    def correct(self):
        """
        Amount that is correct.

        Example:
            >>> ref = 'xxx'
            >>> hyp = 'xxx'
        """
        pass

    @property
    @abc.abstractmethod
    def insertions(self):
        """
        Amount that is inserted.

        Example:
            >>> ref = None
            >>> hyp = 'xxx'
        """
        pass

    @property
    @abc.abstractmethod
    def deletions(self):
        """
        Amount that is deleted.

        Example:
            >>> ref = 'xxx'
            >>> hyp = None
        """
        pass

    @property
    @abc.abstractmethod
    def substitutions(self):
        """
        Amount that is substituted.

        If this stats are representing stats for a specific instance (e.g. occurrence of the word 'hello')
        ``substitutions`` is the amount where the specific instance was substituted with some other instance/event.
        If not it is not necessary to designate which event/instance substitutes which event/instance.

        Example:
            >>> ref = 'xxx'
            >>> hyp = 'yyy'
        """
        pass

    @property
    @abc.abstractmethod
    def substitutions_out(self):
        """
        Amount that is substituted.

        If this stats are representing stats for a specific instance (e.g. occurrence of the word 'hello')
        ``substitutions_out`` is the amount where the specific instance was output,
        when some other event/instance was expected (reference).
        If not it is equal to ``substitutions``.

        Example:
            >>> ref = 'yyy'
            >>> hyp = 'xxx'
        """
        pass

    @property
    def total(self):
        """
        Return the total amount based on the reference system.

        Note:
            Equal to 'self.correct + self.deletions + self.substitutions'
        """
        return self.correct + self.deletions + self.substitutions

    @property
    def false_negatives(self):
        """
        Amount of false negatives (No indication of precence, when it should be present).

        Note:
            Equal to 'self.total - self.correct'
        """
        return self.deletions + self.substitutions

    @property
    def false_positives(self):
        """
        Amount of false positives (Indications of presence, when it is not present).

        Note:
            Equal to `self.insertions + self.substitutions_out`
        """
        return self.insertions + self.substitutions_out

    @property
    def true_positives(self):
        """
        Amount of true positives (Correct indications).

        Note:
             Equal to `self.correct`
        """
        return self.correct

    #
    #   Common Metrics
    #

    @property
    def error_rate(self):
        """ ErrorRate = (substitutions + deletions + insertions) / total """
        total = self.total

        if total <= 0:
            return 0.0

        return (self.substitutions + self.deletions + self.insertions) / total

    @property
    def accuracy(self):
        """ Accuracy = correct / (total + insertions) """
        total = self.total + self.insertions

        if total <= 0:
            return 0.0

        return max(0.0, self.correct / total)

    @property
    def precision(self):
        """ Precision = tp / (fp + tp) """
        total = self.true_positives + self.false_positives

        if total <= 0:
            return 0.0

        return self.true_positives / total

    @property
    def recall(self):
        """ Recall = tp / (fn + tp) """
        if self.total <= 0:
            return 0.0

        return self.correct / self.total

    def f_measure(self, beta=1):
        """
        F-Measure
        see https://en.wikipedia.org/wiki/Precision_and_recall
        """
        prec = self.precision
        rec = self.recall

        return (1 + beta * beta) * (prec * rec) / (beta * beta * prec + rec)
