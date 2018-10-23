from . import utils


class InvariantSegmentAligner(object):
    """
    Create a segment-based alignment so that within every segment the same labels are active.
    So for example as reference we have a label-list as following.

    >>> [   A   ]     [     B     ]    [      A     ]
    >>>                               [      E     ]

    The output of some system (hypothesis) maybe as follows:

    >>> [   Ax  ]     [  Ex ]                           [  Ax ]

    Now the segments returned are created, so every segment represents some time range where the labels are equal.

    >>>         S1      S2   S3    S4    S5       S6      S7   S8
    >>>
    >>> HYP  |   A   |     |  B  |  B  |    |      A     |   |     |
    >>> HYP  |       |     |     |     |    |      E     |   |     |
    >>> REF  |   Ax  |     |  Ex |     |    |            |   |  Ax |
    """

    def align(self, ll_ref, ll_hyp):
        """
        Create segment based alignment.

        Args:
            ll_ref (audiomate.corpus.assets.LabelList): The label-list with reference labels.
            ll_hyp (audiomate.corpus.assets.LabelList): The label-list with hypothesis labels.

        Returns:
            list: A list of Segments.

        Example:
            >>> from audiomate.corpus import assets
            >>>
            >>> ref = assets.LabelList(labels=[
            >>>     assets.Label('a', 0, 3),
            >>>     assets.Label('b', 3, 6),
            >>>     assets.Label('c', 7, 10)
            >>> ])
            >>>
            >>> hyp = assets.LabelList(labels=[
            >>>     assets.Label('a', 0, 3),
            >>>     assets.Label('b', 4, 8),
            >>>     assets.Label('c', 8, 10)
            >>> ])
            >>>
            >>> InvariantSegmentAligner().align(ref, hyp)
            [
                0 - 3 REF: [Label(a, 0, 3)] HYP: [Label(a, 0, 3)]
                3 - 4 REF: [Label(b, 3, 6)] HYP: []
                4 - 6 REF: [Label(b, 3, 6)] HYP: [Label(b, 4, 8)]
                6 - 7 REF: [] HYP: [Label(b, 4, 8)]
                7 - 8 REF: [Label(c, 7, 10)] HYP: [Label(b, 4, 8)]
                8 - 10 REF: [Label(c, 7, 10)] HYP: [Label(c, 8, 10)]
            ]

        """

        refs = InvariantSegmentAligner.set_absolute_end_of_labels(ll_ref)
        hyps = InvariantSegmentAligner.set_absolute_end_of_labels(ll_hyp)

        events = InvariantSegmentAligner.create_event_list(refs, hyps, time_threshold=0.01)

        current_ref = []
        current_hyp = []

        current_start = 0
        segments = []

        # At every event the current ref/hyp lists are updated and a new segment created.
        for index, event in enumerate(events):
            time = event[0]
            sub_events = event[1]

            if index == 0:
                current_start = time

                # Add all labels that are active from the beginning
                for sub_event in sub_events:
                    if sub_event[2] == 0:
                        current_ref.append(sub_event[3])
                    else:
                        current_hyp.append(sub_event[3])
            else:
                new_segment = utils.Segment(current_start, time)
                new_segment.ref = list(current_ref)
                new_segment.hyp = list(current_hyp)
                segments.append(new_segment)

                current_start = time

                # Remove or Add labels to keep track of current active labels
                for sub_event in sub_events:
                    is_start = sub_event[1] == 'S'
                    is_ref = sub_event[2] == 0
                    label = sub_event[3]

                    if is_start:
                        if is_ref:
                            current_ref.append(label)
                        else:
                            current_hyp.append(label)
                    else:
                        if is_ref:
                            current_ref.remove(label)
                        else:
                            current_hyp.remove(label)

        return segments

    @staticmethod
    def create_event_list(ll_ref, ll_hyp, time_threshold=0.01):
        """
        Create an event list of all labels.

        Arguments:
            ll_ref (LabelList): Reference labels.
            ll_hyp (LabelList): Hypothesis labels.
            time_threshold (float): If two event times are closer than this threshold the time of the
                                    earlier event is used for both events.

        Returns:
            list: List of list of tuples. Every tuple contains a time, type (start or end), ll_index (ref/hyp) and
            the label which is responsible for the event. It is sorted ascending by time.
        """
        events = []

        for label in ll_ref:
            events.append((label.start, 'S', 0, label))
            events.append((label.end, 'E', 0, label))

        for label in ll_hyp:
            events.append((label.start, 'S', 1, label))
            events.append((label.end, 'E', 1, label))

        events = sorted(events, key=lambda x: x[0])
        time_grouped = []
        current_group = (events[0][0], [events[0]])

        for i in range(1, len(events)):
            if abs(events[i][0] - events[i - 1][0]) < time_threshold:
                current_group[1].append(events[i])
            else:
                time_grouped.append(current_group)
                current_group = (events[i][0], [events[i]])

        time_grouped.append(current_group)

        return time_grouped

    @staticmethod
    def set_absolute_end_of_labels(label_list):
        """
        If there are any labels where the end is defined as -1 (end of utterance),
        set the concrete time.

        Arguments:
            label_list (LabelList): The label-list to process.
        """

        for label in sorted(label_list, key=lambda x: x.start):
            if label.end <= label.start:
                raise ValueError('Label-end {} is smaller than label-start {}!'.format(label.end, label.start))

            if label.value is not '###############':
                if label.end == -1:
                    label.end = label.label_list.utterance.end

                if label.end == -1:
                    label.end = label.label_list.utterance.file.duration - label.label_list.utterance.start

        return label_list
