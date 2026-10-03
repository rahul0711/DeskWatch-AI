from app.tracking.text_tracker import TextReading, TextTracker


def _reading(bbox, text, confidence=0.8, moving=False):
    return TextReading(bbox=bbox, text=text, confidence=confidence, quad=None, moving=moving)


def test_same_position_keeps_same_id():
    tracker = TextTracker()
    t1 = tracker.update([_reading((10, 10, 110, 40), "EXIT")])
    t2 = tracker.update([_reading((12, 11, 112, 41), "EXIT")])
    assert t1[0].track_id == t2[0].track_id


def test_new_region_gets_new_id():
    tracker = TextTracker()
    t1 = tracker.update([_reading((10, 10, 110, 40), "EXIT")])
    t2 = tracker.update([_reading((10, 10, 110, 40), "EXIT"), _reading((500, 500, 600, 530), "PUSH")])
    ids = {t.track_id for t in t2}
    assert t1[0].track_id in ids
    assert len(ids) == 2


def test_majority_vote_wins_over_single_misread():
    tracker = TextTracker()
    for text in ("EXIT", "EXIT", "EX1T", "EXIT"):
        result = tracker.update([_reading((10, 10, 110, 40), text)])
    text, conf, votes = result[0].voted()
    # "EX1T" is similar enough to "EXIT" (single-char OCR noise) to land in
    # the same voting cluster -- that's the point of the similarity match.
    assert text == "EXIT"
    assert votes == 4
    assert conf > 0


def test_dissimilar_readings_do_not_merge():
    tracker = TextTracker()
    for text in ("EXIT", "PUSH", "EXIT"):
        result = tracker.update([_reading((10, 10, 110, 40), text)])
    text, conf, votes = result[0].voted()
    assert text == "EXIT"
    assert votes == 2


def test_confirmation_requires_minimum_votes():
    tracker = TextTracker()
    result = tracker.update([_reading((10, 10, 110, 40), "EXIT")])
    _, _, votes = result[0].voted()
    assert votes < tracker.min_votes_to_confirm


def test_disappearing_region_frees_up_gracefully():
    tracker = TextTracker(max_missed_frames=1)
    tracker.update([_reading((10, 10, 110, 40), "EXIT")])
    result = tracker.update([])
    assert result == []


def test_moving_flag_latches_true_once_seen_moving():
    tracker = TextTracker()
    tracker.update([_reading((10, 10, 110, 40), "CARD", moving=True)])
    result = tracker.update([_reading((11, 10, 111, 40), "CARD", moving=False)])
    assert result[0].ever_moving is True
