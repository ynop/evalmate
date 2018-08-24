import abc

import audiomate
from audiomate.corpus import assets
from jinja2 import Environment, PackageLoader, select_autoescape

from . import outcome

env = Environment(
    loader=PackageLoader('evalmate.tasks', 'report_templates'),
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
        if template is None:
            template = self.default_template

        with open(path, 'w') as f:
            template = self._load_template(template)
            f.write(template.render(**self.template_data))

    def _load_template(self, name):
        return env.get_template('{}.txt'.format(name))


class Evaluator(abc.ABC):
    """
    Base class for a evaluator.

    Provides methods for reading outcomes in different ways.
    The evaluator for a specific class then has to implement ``do_evaluate``,
    which performs the evaluation on ref and hyp outcome.
    """

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

        if isinstance(ref, assets.LabelList) and isinstance(hyp, assets.LabelList):
            return self.evaluate_label_lists(ref, hyp)

        if isinstance(ref, audiomate.Corpus) and isinstance(hyp, dict):
            return self.evaluate_label_lists_against_corpus(ref, hyp, label_list_idx=label_list_idx)

        raise ValueError('Invalid arguments!')

    def evaluate_label_lists(self, ll_ref, ll_hyp):
        """
        Create Evaluation for ref and hyp label-list.

        Arguments:
            ref (LabelList): A label-list.
            hyp (LabelList): A label-list.

        Returns:
            Evaluation: The evaluation results.
        """

        ref_outcome = outcome.Outcome(label_lists={'0': ll_ref})
        hyp_outcome = outcome.Outcome(label_lists={'0': ll_hyp})

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

        return self.evaluate(ref_outcome, hyp_outcome)
