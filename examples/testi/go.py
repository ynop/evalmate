import os
import evalmate
import audiomate
from audiomate.formats import audacity


corpus = audiomate.Corpus.load('ref_arg')

hyps = {}

for utt_id in corpus.utterances.keys():
    ll = audacity.read_label_list(
        os.path.join('hyp_arg', '{}.txt'.format(utt_id)))
    hyps[utt_id] = ll

ev = evalmate.ClassificationEvaluator()
result = ev.evaluate_label_lists_against_corpus(corpus, hyps,
                                                label_list_idx='flat')
result.write_report('result_arg.txt', template='classification_detail')
