import abc

import audiomate
from audiomate import annotations
from jinja2 import Environment, PackageLoader, select_autoescape

from . import outcome

env = Environment(
    loader=PackageLoader('evalmate.evaluator', 'report_templates'),
    autoescape=select_autoescape(['html', 'xml'])
)


class Evaluation(abc.ABC):
    """
    Base class for evaluation results.

    Attributes:
        ref_outcome (Outcome): The outcome of the ground-truth/reference.
        hyp_outcome (Outcome): The outcome of the system-output/hypothesis.
    """

    def __init__(self, ref_outcome, hyp_outcome):
        self.ref_outcome = ref_outcome
        self.hyp_outcome = hyp_outcome

    @property
    @abc.abstractmethod
    def template_data(self):
        """ Return a dictionary that contains objects/values to use in the rendering template. """
        return {}

    @property
    def default_template(cls):
        return 'default'

    def write_report(self, path, template=None):
        """
        Write the report to the given path.

        Args:
            path (str): Path to write the report to.
            template (str): Name of the Jinja2 template to use. If None, the ``default_template()`` is used.
                            All available templates are in the ``report_templates`` folder.
        """
        with open(path, 'w') as f:
            f.write(self.get_report(template=template))

    def get_report(self, template=None):
        """
        Generate and return a report.

        Args:
            template (str): Name of the Jinja2 template to use. If None, the ``default_template()`` is used.
                            All available templates are in the ``report_templates`` folder.

        Returns:
            str: The rendered report.
        """
        if template is None:
            template = self.default_template

        template = self._load_template(template)
        return template.render(**self.template_data)

    def _load_template(self, name):
        return env.get_template('{}.txt'.format(name))


class Evaluator(abc.ABC):
    """
    Base class for a evaluator.

    Provides methods for reading outcomes in different ways.
    The evaluator for a specific class then has to implement ``do_evaluate``,
    which performs the evaluation on ref and hyp outcome.
    """

    DEFAULT_UTT_IDX = 'noname'

    @classmethod
    @abc.abstractmethod
    def default_label_list_idx(cls):
        """ Define the default label-lists which is used when reading a corpus. """
        return 'default'

    @abc.abstractmethod
    def do_evaluate(self, ref, hyp):
        """
        Create the evaluation result of the given hypothesis compared to the given reference (ground truth).

        Arguments:
            ref (Outcome): The ground-truth/reference outcome.
            hyp (Outcome): The system-output/hypothesis outcome.

        Returns:
            Evaluation: The evaluation results.
        """
        pass

    def evaluate(self, ref, hyp, label_list_idx=None):
        """
        Create the evaluation result of the given hypothesis compared to the given reference (ground truth).
        There are different possibilities of input:

        * ref = Outcome / hyp = Outcome: Both ref and hyp are `Outcome` instances.
          See ``do_evaluate``
        * ref = Corpus / hyp = dict: The dict contains label-lists which are compared against the corpus.
          See ``evaluate_label_lists_against_corpus``
        * ref = LabelList / hyp = LabelList: Ref label-list is compared against the other.
          See ``evaluate_label_lists``

        Arguments:
            ref (LabelList, Corpus): A label-list, a corpus.
            hyp (LabelList, dict): A label-list, a dict.
            label_list_idx (str): The label-list to use when reading from a corpus.

        Returns:
            Evaluation: The evaluation results.
        """

        if isinstance(ref, outcome.Outcome) and isinstance(hyp, outcome.Outcome):
            return self.do_evaluate(ref, hyp)

        if isinstance(ref, annotations.LabelList) and isinstance(hyp, annotations.LabelList):
            return self.evaluate_label_lists(ref, hyp)

        if isinstance(ref, audiomate.Corpus) and isinstance(hyp, dict):
            return self.evaluate_label_lists_against_corpus(ref, hyp, label_list_idx=label_list_idx)

        raise ValueError('Invalid arguments!')

    def evaluate_label_lists(self, ll_ref, ll_hyp, duration=None):
        """
        Create Evaluation for ref and hyp label-list.
        If the duration is not provided some metrics cannot be used.

        Arguments:
            ref (LabelList): A label-list.
            hyp (LabelList): A label-list.
            duration (float): The duration of the utterance, that belongs to the label-lists.

        Returns:
            Evaluation: The evaluation results.
        """

        durations = None

        if duration is not None:
            durations = {self.DEFAULT_UTT_IDX: duration}

        ref_outcome = outcome.Outcome(label_lists={self.DEFAULT_UTT_IDX: ll_ref}, utterance_durations=durations)
        hyp_outcome = outcome.Outcome(label_lists={self.DEFAULT_UTT_IDX: ll_hyp}, utterance_durations=durations)

        return self.evaluate(ref_outcome, hyp_outcome)

    def evaluate_label_lists_against_corpus(self, corpus, label_lists, label_list_idx=None):
        """
        Create Evaluation for the given corpus.

        Arguments:
            corpus (Corpus): A corpus containing the reference label-lists.
            label_lists (Dict): A dictionary containing label-lists with the utterance-idx as key.
                                The utterance-idx is used to find the corresponding reference label-list in the corpus.
            label_list_idx (str): The idx of the label-lists to use as reference from the corpus.
                                  If None, `cls.default_label_list_idx` is used.

        Returns:
            Evaluation: The evaluation results.
        """
        label_list_idx = label_list_idx or self.default_label_list_idx()

        ref_outcome = outcome.Outcome()
        hyp_outcome = outcome.Outcome()

        for utterance in corpus.utterances.values():
            ll_ref = utterance.label_lists[label_list_idx]

            if utterance.idx not in label_lists:
                raise ValueError('There is no hypothesis label-list with idx {}'.format(utterance.idx))

            ll_hyp = label_lists[utterance.idx]

            ref_outcome.label_lists[utterance.idx] = ll_ref
            hyp_outcome.label_lists[utterance.idx] = ll_hyp

            ref_outcome.utterance_durations[utterance.idx] = utterance.duration
            hyp_outcome.utterance_durations[utterance.idx] = utterance.duration

        return self.evaluate(ref_outcome, hyp_outcome)
