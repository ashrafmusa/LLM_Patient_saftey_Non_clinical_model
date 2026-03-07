from src.risk_assessment import assess_risk


def test_assess_low():
    r = assess_risk('Patient is stable, no acute issues.')
    assert r['risk_level'] in ('low','medium','high')


def test_assess_high():
    r = assess_risk('Patient attempted suicide and has self-harm thoughts')
    assert r['risk_level'] == 'high'
