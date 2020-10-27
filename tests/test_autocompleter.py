""" Tests to validate that the autocompleter functions correctly."""

import autocompleter
import pytest
import timeit


MAX_COMPLETION_TIME_MS=50


@pytest.fixture(scope="module")
def auto_completer():
    """Creates an autocompleter object used in the following tests"""
    completer = autocompleter.Autocompleter()
    completer.fit("tests/data/test_conversations.json", company_group_id_filter=[50001], verbose=False )
    return completer


@pytest.mark.parametrize("prefix, max_number_words, max_number_sentence, expected_output",
                         [("How can i", 10, 2, ['How can I assist you with today?', 'How can I help you with today?']),
                          ("How can i", 5, 2, ['How can I assist you', 'How can I help you']),
                          ("How can i", 5, 1, ['How can I assist you']),
                          ("Are y", 5, 1, ['Are you aware of this']),
                          ("How", 5, 1, []),
                          ("", 5, 1, [])])
def test_generate_completions(auto_completer, prefix, max_number_words, max_number_sentence, expected_output):
    """Test some completion generations based on known results on the training conversations used"""
    start = timeit.timeit()
    completions = auto_completer.generate_completions(prefix, max_number_words, max_number_sentence)
    end = timeit.timeit()
    assert completions == expected_output
    assert (end - start) < MAX_COMPLETION_TIME_MS


def test_non_fitted_autocompleter(auto_completer):
    """Test class method outputs on a non fitted autocompleter"""
    non_fitted_completer = autocompleter.Autocompleter()

    # Test get_next_word on non fitted completer
    non_fitted_completion = non_fitted_completer.get_next_word(["how", "can", "i"])
    fitted_completion = auto_completer.get_next_word(["how", "can", "i"])
    assert non_fitted_completion == [] and fitted_completion != []

    # Test get_sentence on non fitted completer
    non_fitted_completion = non_fitted_completer.get_sentence(["how", "can", "i"])
    fitted_completion = auto_completer.get_sentence(["how", "can", "i"])
    assert non_fitted_completion == [] and fitted_completion != []

    # Test get_sentence on non fitted completer
    non_fitted_completion = non_fitted_completer.generate_completions("how can i")
    fitted_completion = auto_completer.generate_completions("how can i")
    assert non_fitted_completion == [] and fitted_completion != []


@pytest.mark.parametrize("prefix, top_n, start_string, expected_output",
                         [(["can", "i"], 2, "", ['assist', 'help']),
                          (["can", "i"], 2, "he", ['help']),
                          (["can", "i"], 1, "", ['assist']),
                          (["how"], 4, "", ['to', 'long', 'may', 'much']),
                          (["how"], 4, "m", ['may', 'much']),
                          (["how"], 4, "h", [])])
def test_get_next_word(auto_completer, prefix, top_n, start_string, expected_output):
    """Test some completion generations based on known results on the training conversations used"""
    completions = auto_completer.get_next_word(prefix, top_n, start_string)
    assert completions == expected_output


@pytest.mark.parametrize("prefix, max_sentence_length, expected_output",
                         [(["how", "can", "i"], 4, ['how', 'can', 'i', 'assist']),
                          (["how", "can", "i"], 9, ['how', 'can', 'i', 'assist', 'you', 'with', 'today', '?']),
                          (["how", "can", "i"], 12, ['how', 'can', 'i', 'assist', 'you', 'with', 'today', '?']),
                          (["what", "is", "your"], 2, ["what", "is", "your"]),
                          (["what", "is", "your"], 6, ['what', 'is', 'your', 'account', 'number', '?'])])
def test_get_sentence(auto_completer, prefix, max_sentence_length, expected_output):
    """Test some completion generations based on known results on the training conversations used"""
    completions = auto_completer.get_sentence(prefix, max_sentence_length)
    assert completions == expected_output


@pytest.mark.parametrize("word, count_threshold, expected_output",
                         [("your", 10, True), ("your", 1000, False), ("blabla", 1, False), ("i", 1000, True)])
def test_in_vocabulary(auto_completer, word, count_threshold, expected_output):
    """Test some completion generations based on known results on the training conversations used"""
    completions = auto_completer.in_vocabulary(word, count_threshold)
    assert completions == expected_output
