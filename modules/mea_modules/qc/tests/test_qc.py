import mea_modules.qc as qc


def test_qc_config_defaults():
    config = qc.QCConfig()
    assert config.highpass_hz == 300.0
    assert config.mad_threshold == 5.0
    assert config.activity_min_rate_hz == 0.1
    assert config.dead_well_frac == 0.5
