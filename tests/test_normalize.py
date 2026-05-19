from canary_sk.normalize import normalize_for_wer, normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("Ahoj  svět  !") == "Ahoj svět !"


def test_normalize_text_keeps_punctuation():
    assert normalize_text("Ahoj, svet!") == "Ahoj, svet!"


def test_normalize_text_nfc():
    # café as NFD (e + combining accent) must become NFC (é)
    nfd = "café"
    assert normalize_text(nfd) == "café"


def test_normalize_text_nbsp():
    assert normalize_text("hello world") == "hello world"


def test_normalize_for_wer_lowercase():
    assert normalize_for_wer("Ahoj Svet") == "ahoj svet"


def test_normalize_for_wer_strips_punctuation():
    assert normalize_for_wer("Ahoj, Svet!") == "ahoj svet"


def test_normalize_for_wer_keeps_slovak_chars():
    assert normalize_for_wer("Čo je to?") == "čo je to"


def test_normalize_for_wer_collapses_whitespace():
    assert normalize_for_wer("  Čo  je  to?") == "čo je to"
