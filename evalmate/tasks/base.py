import abc

from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader('evalmate.tasks', 'report_templates'),
    autoescape=select_autoescape(['html', 'xml'])
)


class Evaluation(abc.ABC):

    @property
    @abc.abstractmethod
    def data(self):
        pass

    @property
    def default_template(cls):
        return 'default'

    def write_report(self, path, template=None):
        if template is None:
            template = self.default_template

        with open(path, 'w') as f:
            template = self._load_template(template)
            f.write(template.render(data=self.data))

    def _load_template(self, name):
        return env.get_template('{}.txt'.format(name))


class Evaluator(abc.ABC):

    @abc.abstractmethod
    def evaluate(self, ref, hyp):
        """
        Create the evaluation result of the given hypothesis compared to the given reference (ground truth).

        Returns:
            Evaluation: The evaluation results.
        """
        pass
