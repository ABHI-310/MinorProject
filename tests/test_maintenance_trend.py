from backend.maintenance import quality_trend


def rows(counts):
    return [dict(count=n, boards_inspected=100, model_type='k500_coco', checkpoint_sha256='a', confidence_cutoff=.25) for n in counts]


def test_increase_triggers_review_and_equal_rate_does_not():
    assert quality_trend(rows([4,4,4,1,1,1]))['alert']
    assert not quality_trend(rows([1,1,1,1,1,1]))['alert']


def test_exposure_normalization_and_no_history():
    history=rows([4,4,4,1,1,1])
    for r in history[:3]: r['boards_inspected']=400
    assert not quality_trend(history)['alert']
    assert quality_trend([])['status']=='insufficient_history'
    history[0]['boards_inspected']=None
    assert quality_trend(history)['status']=='exposure_required'


def test_settings_changes_block_comparison():
    history=rows([4,4,4,1,1,1]);history[0]['confidence_cutoff']=.5
    assert quality_trend(history)['status']=='settings_changed'


def test_synthetic_demo_is_explicit_and_triggers_review():
    from backend.maintenance import synthetic_review_demo
    demo=synthetic_review_demo()
    assert demo['synthetic'] and not demo['stored']
    assert demo['trend']['synthetic'] and demo['trend']['alert']
    assert demo['trend']['recent_rate_per_100_boards']==4
    assert demo['trend']['previous_rate_per_100_boards']==1
