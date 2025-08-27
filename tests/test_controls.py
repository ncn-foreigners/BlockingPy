"""Tests for the InputValidator class in blockingpy.helper_functions."""

import pytest

from blockingpy.helper_functions import InputValidator


def test_invalid_encoder():
    """
    Should raise ValueError when encoder is not one of the allowed options.
    """
    bad = {"encoder": "bad"}
    with pytest.raises(ValueError) as exc:
        InputValidator.validate_controls_txt(bad)
    assert "Unknown encoder" in str(exc.value)


@pytest.mark.parametrize("enc", ["shingle", "embedding"])
def test_valid_encoder_options(enc):
    """
    Should not raise for valid encoder.
    """
    valid = {"encoder": enc}
    InputValidator.validate_controls_txt(valid)


def test_unknown_subkey_in_section_shingles():
    """
    Should raise ValueError if a typo or unknown key is present in the chosen encoder section.
    """
    bad = {"encoder": "shingle", "shingle": {"n_shinhless": 5}}
    with pytest.raises(ValueError) as exc:
        InputValidator.validate_controls_txt(bad)
    assert "Unknown keys in control_txt['shingle']" in str(exc.value)


def test_unknown_subkey_in_section_embedd():
    """
    Should raise ValueError if a typo or unknown key is present in the chosen encoder section.
    """
    bad = {"encoder": "embedding", "embedding": {"show_progresss_bar": False}}
    with pytest.raises(ValueError) as exc:
        InputValidator.validate_controls_txt(bad)
    assert "Unknown keys in control_txt['embedding']" in str(exc.value)
