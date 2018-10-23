"""
This module implements the top-level functionality for performing the evaluation for the different tasks.
For every task there is an Evaluator (extends :py:class:`Evaluator`) and an Evaluation (extends :py:class:`Evaluation`.
The Evaluator is the is class responsible to perform the evaluation and the Evaluation is the output,
which contains the aligned labels/segments and depending on the task further data like word confusions.

.. currentmodule:: evalmate.evaluator

Base
----

.. autoclass:: Evaluation
   :members:

.. autoclass:: Evaluator
   :members:

Outcome
-------

.. autoclass:: Outcome
   :members:

.. autoclass:: LabelSet
   :members:

Segment
-------

.. autoclass:: SegmentEvaluation
   :members:

.. autoclass:: SegmentEvaluator
   :members:

Event
-----

.. autoclass:: EventEvaluation
   :members:

.. autoclass:: EventEvaluator
   :members:

KWS
---

.. autoclass:: KWSEvaluation
   :members:

.. autoclass:: KWSEvaluator
   :members:

ASR
---

.. autoclass:: ASREvaluation
   :members:

.. autoclass:: ASREvaluator
   :members:

"""

from .outcome import Outcome  # noqa: F401
from .outcome import LabelSet  # noqa: F401

from .evaluator import Evaluation  # noqa: F401
from .evaluator import Evaluator  # noqa: F401

from .event import EventEvaluator  # noqa: F401
from .event import EventEvaluation  # noqa: F401

from .segment import SegmentEvaluator  # noqa: F401
from .segment import SegmentEvaluation  # noqa: F401

from .kws import KWSEvaluation  # noqa: F401
from .kws import KWSEvaluator  # noqa: F401

from .asr import ASREvaluation  # noqa: F401
from .asr import ASREvaluator  # noqa: F401
