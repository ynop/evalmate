import os

import audiomate
from audiomate.corpus import assets
from audiomate.formats import audacity

import pytest

from tests import resources


@pytest.fixture
def kws_ref_and_hyp_label_list():
    """
    Sample output of a kws system, consisting of a reference and hypothesis label-list.
    """

    ll_ref = assets.LabelList(labels=[
        assets.Label('up', start=5.28, end=5.99),
        assets.Label('down', start=10.35, end=11.12),
        assets.Label('right', start=20.87, end=22.01),
        assets.Label('up', start=33.00, end=33.4),
        assets.Label('up', start=33.4, end=33.8),
        assets.Label('down', start=39.28, end=40.0)
    ])

    ll_hyp = assets.LabelList(labels=[
        assets.Label('up', start=5.20, end=5.88),
        assets.Label('right', start=10.30, end=11.08),
        assets.Label('up', start=32.00, end=32.5),
        assets.Label('up', start=34.2, end=34.8),
        assets.Label('left', start=39.3, end=39.9),
        assets.Label('down', start=39.27, end=40.01)
    ])

    return ll_ref, ll_hyp


@pytest.fixture
def kws_ref_corpus_and_hyp_labels():
    """
    Sample corpus and hypothesis label-lists from a kws task.
    """
    ref_path = os.path.join(os.path.dirname(resources.__file__), 'kws', 'ref_corpus')
    hyp_path = os.path.join(os.path.dirname(resources.__file__), 'kws', 'hyp')
    corpus = audiomate.Corpus.load(ref_path)
    hyps = {}

    for utt_id in corpus.utterances.keys():
        ll = audacity.read_label_list(os.path.join(hyp_path, '{}.txt'.format(utt_id)))
        hyps[utt_id] = ll

    return corpus, hyps


@pytest.fixture
def classification_ref_and_hyp_label_list():
    """
    Sample output of a classification system, consisting of a reference and hypothesis label-list.
    """

    ll_ref = assets.LabelList(labels=[
        assets.Label('music', start=0, end=5),
        assets.Label('speech', start=5, end=11),
        assets.Label('mix', start=11, end=14),
        assets.Label('speech', start=14, end=19)
    ])

    ll_hyp = assets.LabelList(labels=[
        assets.Label('music', start=0, end=4),
        assets.Label('speech', start=4, end=6),
        assets.Label('mix', start=8, end=16),
        assets.Label('speech', start=16, end=21)
    ])

    return ll_ref, ll_hyp


@pytest.fixture
def classification_ref_corpus_and_hyp_labels():
    """
    Sample corpus and hypothesis label-lists from a classification task.
    """
    ref_path = os.path.join(os.path.dirname(resources.__file__), 'classification', 'ref_corpus')
    hyp_path = os.path.join(os.path.dirname(resources.__file__), 'classification', 'hyp')
    corpus = audiomate.Corpus.load(ref_path)
    hyps = {}

    for utt_id in corpus.utterances.keys():
        ll = audacity.read_label_list(os.path.join(hyp_path, '{}.txt'.format(utt_id)))
        hyps[utt_id] = ll

    return corpus, hyps
