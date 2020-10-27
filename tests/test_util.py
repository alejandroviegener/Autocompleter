""" Tests to validate the util functions."""

import util
import json
from nltk import tokenize
import pytest

SPELL_CORRECTOR_ACCURACY=70
TRUE_CASE_ACCURACY=80
ENTITY_ENCODER_ACCURACY=75


def test_get_operator_messages():
    """Test over a small conversations sample"""
    json_data = json.load(open("tests/data/test_get_messages_data.json"))

    company_1_data = util.get_operator_messages(json_data, [1])
    company_2_data = util.get_operator_messages(json_data, [2])
    company_3_data = util.get_operator_messages(json_data, [3])
    all_companies_data = util.get_operator_messages(json_data)

    expected_company_1_data = ["Hi! I placed an order on your website and I can't find the tracking number."
                               " Can you help me find out where my package is?",
                               "I think I used my email address to log in."]
    expected_company_2_data = ['My battery exploded!', "It's on fire, it's melting the carpet!", 'What should I do!']
    expected_company_3_data = ["I'm interested in upgrading my plan.", 'Can you tell me a bit about Prime?',
                               'My friend has it, and it seems like a great deal']
    expected_all_companies_data = expected_company_1_data + expected_company_2_data + expected_company_3_data

    assert company_1_data == expected_company_1_data
    assert company_2_data == expected_company_2_data
    assert company_3_data == expected_company_3_data
    assert all_companies_data == expected_all_companies_data


def test_format_and_clean():
    input_text = "This is a test <text>, to test the format_clean function."
    expected_output = "this is a test to test the format_clean function"
    assert util.format_and_clean(input_text) == expected_output


def test_sentence_tokenize():
    input_documents = ["This is a sentence of document 1. This is another sentence", "This is a second document"]
    expected_output = ["This is a sentence of document 1.", "This is another sentence", "This is a second document"]
    assert util.sentence_tokenize(input_documents) == expected_output


def test_spell_correction():
    """Spell corrector must not be perfect, test accuracy over a test text"""
    input_text = ["This is a sequncte of texts", "With erors", "to testt", "teh spell corector",
                  "The idea is to cont", "the presentage", "of correctt", "answerss", "Ths text",
                  "shoud probably", "be longerr"]
    expected_output = ["This is a sequence of texts", "With errors", "to test", "the spell corrector",
                       "The idea is to count", "the percentage", "of correct", "answers", "This text",
                       "should probably", "be longer"]

    output = [util.spell_correction(text) for text in input_text]
    total = len(input_text)
    differences = len([text1 for text1, text2 in zip(output, expected_output)  if text1 != text2 ])
    accuracy = 100*(total - differences)/total

    assert accuracy > SPELL_CORRECTOR_ACCURACY


def test_true_case():
    """True case operation must not be perfect, test accuracy over a test text"""
    input_text = ["this is a sequence of texts.", "i have never been to india.", "my name is alejandro",
                  "i would like to have a bmw car.", "how can i test this?", "again, this sequence should be longer"]
    expected_output = ["This is a sequence of texts.", "I have never been to India.", "My name is Alejandro",
                       "I would like to have a BMW car.", "How can I test this?",
                       "Again, this sequence should be longer"]

    output = [util.get_true_case(text) for text in input_text]
    total = len(input_text)
    differences = len([text1 for text1, text2 in zip(output, expected_output)  if text1 != text2 ])
    accuracy = 100*(total - differences)/total

    assert accuracy > TRUE_CASE_ACCURACY


def test_encode_entities():
    """Entity encoder must not be perfect, test accuracy over a test text"""
    input_text = ["my name is james", "no entities here.", "i work at mbw",
                  "my brother's name is john", "and my sister's name is mary",
                  "this text should be longer"]
    expected_output = ['my name is __PER__', 'no entities here.', 'i work at __ORG__', "my brother's name is __PER__",
                       "and my sister's name is __PER__", 'this text should be longer']

    output = [util.encode_entities(text) for text in input_text]
    total = len(input_text)
    differences = len([text1 for text1, text2 in zip(output, expected_output)  if text1 != text2 ])
    accuracy = 100*(total - differences)/total

    assert accuracy > ENTITY_ENCODER_ACCURACY
