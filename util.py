"""Utils functions used for text pre-processing and json file reading.

"""

import autocorrect
import re
import en_core_web_sm
from nltk import tokenize
from nltk.lm import preprocessing
import truecase

util_nlp = en_core_web_sm.load()
util_spell_corrector = autocorrect.Speller(lang='en')


def get_operator_messages(json_conversation_structure, company_group_id_filter=None):
    """Gets the messages of the operator found in the json structure.

    Args:
        json_conversation_structure: json structure containing the operator messages.
            (...the json file structure should also be explained here...)
        company_group_id_filter: list of company group ids to consider in the json file. If None,
            then all companies are considered

    Returns:
        A list containing the operator messages. Every list element (message) can contain more than one sentence. The
        returned value is NOT sentence tokenized.
    """

    operator_messages = []
    for issue in json_conversation_structure["Issues"]:
        if (company_group_id_filter is None) or (issue["CompanyGroupId"] in company_group_id_filter):
            for message in issue["Messages"]:
                if not message["IsFromCustomer"]:
                    operator_messages.append(message["Text"])

    return operator_messages


def format_and_clean(text):
    """Transform to lower case, cleans some symbols from the text, i.e., <>.,"

    Args:
        text: input text to apply the transformation.

    Returns:
        The cleaned and formatted text
    """

    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub('[.,"]', '', text)
    text = re.sub(' +', ' ', text)
    return text


def spell_correction(text):
    """Spell correct the input text"""
    sentence = util_spell_corrector.autocorrect_sentence(text)
    return sentence


def sentence_tokenize(documents):
    """Returns a list of tokenized sentences

    Args:
        documents: list of texts. Each element of the list can contain several sentences, e.g.:
            ["This is the first document. Just another sentence.", "Second document.", "Hey! Last but not least."]

    Returns:
        Returns a list of tokenized sentences:
            ["This is the first document", "Just another sentence", "Second document", "Hey!", "Last but not least"]
    """

    sentences = [tokenize.sent_tokenize(message) for message in  documents]
    sentences = list(preprocessing.flatten(sentences))
    return sentences


def encode_entities(document, person_label="__PER__", organization_label="__ORG__"):
    """Recognize entities on the document and replace them with the specified labels
        See https://spacy.io/api/annotation#section-named-entities for all supported entities recognition

    Args:
        document: input text
        xyz_label: the label that is going to be used to replace the entity xyz
            (...other entities could also be added such as date, time, money, etc)

    Returns:
        The text with the entities replaced with the corresponding labels
    """

    # Get entities from current document
    nlp_document = util_nlp(document)
    entities = [(entity.text, entity.label_) for entity in nlp_document.ents]

    # Replace entities with corresponding labels
    for text, label in entities:
        if label == "PERSON":
            document = document.replace(text, person_label)
        if label == "ORG":
            document = document.replace(text, organization_label)

    # Other common standard entities replacement
    document = document.replace("Snoopy", person_label)
    document = document.replace("snoopy", person_label)

    return document


def get_true_case(text):
    """Transform a text to its true case"""
    return truecase.get_true_case(text)
