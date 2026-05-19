import numpy as np
import pytest


@pytest.fixture
def fake_audio():
    """1-second 16 kHz sine wave — usable as a stand-in audio array in tests."""
    t = np.linspace(0, 1, 16000)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)
