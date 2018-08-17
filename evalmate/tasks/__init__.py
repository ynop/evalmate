"""
This module implements the top-level functionality for performing the evaluation for the different tasks.
For every task there is an Evaluator (extends :py:class:`Evaluator`) and an Evaluation (extends :py:class:`Evaluation`.
The Evaluator is the is class responsible to perform the evaluation and the Evaluation is the output,
which contains the aligned labels/segments and depending on the task further data like word confusions.

.. currentmodule:: evalmate.tasks

Base
----

.. autoclass:: Evaluation
   :members:

.. autoclass:: Evaluator
   :members:

Classification
--------------

.. autoclass:: ClassificationEvaluation
   :members:

.. autoclass:: ClassificationEvaluator
   :members:

KWS
---

.. autoclass:: KWSEvaluation
   :members:

.. autoclass:: KWSEvaluator
   :members:


"""

from .base import Evaluation  # noqa: F401
from .base import Evaluator  # noqa: F401

from .classification import ClassificationEvaluator  # noqa: F401
from .classification import ClassificationEvaluation  # noqa: F401

from .kws import KWSEvaluation  # noqa: F401
from .kws import KWSEvaluator  # noqa: F401
