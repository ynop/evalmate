Changelog
=========

Next Version
------------

v0.3.0
------

**Breaking changes**

* Refactoring of all elements, so that it is more obvious which aligner is used for which evaluator and confusion.

**New Features**

* Introduced False Rejection Rate, False Alarm Rate, Term-Weight Value for the Keyword Spotting task.

* Evaluator for the Automatic Speech Recognition Task :class:`evalmate.evaluator.ASREvaluator`.

v0.2.0
------

**New Features**

* Introduced :class:`evalmate.evaluator.Outcome` to have a common input structure for reference and hypothesis.

* With :class:`evalmate.evaluator.LabelSet` more statistics on reference and hypothesis can be computed.
  Label-Sets are created via :class:`evalmate.evaluator.Outcome` class.

v0.1.0
------

Initial release
