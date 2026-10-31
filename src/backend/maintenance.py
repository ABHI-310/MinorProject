"""Descriptive output-quality monitoring; thresholds are illustrative demo rules."""


def quality_trend(history):
    result = {'status': 'insufficient_history', 'alert': False, 'synthetic': False,
              'note': 'Review rule only; not a machine-failure prediction.',
              'rule': 'Three recent vs three previous comparable inspections; alert if rate doubles and rises by at least 2 detections per 100 boards.'}
    if len(history) < 6:
        return result
    window = history[:6]  # newest first
    if any(not row.get('boards_inspected') for row in window):
        return {**result, 'status': 'exposure_required'}
    def profile(row):
        return (row.get('model_type'), row.get('checkpoint_sha256'), row.get('confidence_cutoff'))
    if len({profile(row) for row in window}) != 1:
        return {**result, 'status': 'settings_changed'}
    def rate(rows):
        return 100 * sum(row['count'] for row in rows) / sum(row['boards_inspected'] for row in rows)
    recent, previous = rate(window[:3]), rate(window[3:])
    alert = recent >= 2 * previous and recent - previous >= 2
    return {**result, 'status': 'review_required' if alert else 'no_rule_trigger', 'alert': alert,
            'recent_rate_per_100_boards': recent, 'previous_rate_per_100_boards': previous,
            'inspection_count': 6, 'model_type': window[0]['model_type']}


def synthetic_review_demo():
    """Six fabricated equal-exposure observations, kept outside inspection storage."""
    history = [dict(count=count, boards_inspected=100, model_type='synthetic_demo',
                    checkpoint_sha256=None, confidence_cutoff=None)
               for count in [4, 4, 4, 1, 1, 1]]
    trend = quality_trend(history)
    trend['synthetic'] = True
    return {'synthetic': True, 'stored': False,
            'description': 'Fabricated scenario: 1 detection per 100 boards rises to 4. No model inference or real machine measurements.',
            'inspections_oldest_first': list(reversed(history)), 'trend': trend}
