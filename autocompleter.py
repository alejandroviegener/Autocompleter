"""Autocompleter class that generates sentences for a prefix string based on a trained corpus.

This implementation was inspired in: https://pypi.org/project/autocomplete/ and
https://github.com/diegoo/mineria_de_texto/blob/master/clase.7/5_Language_Models.ipynb

"""

import json
import pickle
import nltk
import util
from nltk.lm import counter
from nltk import tokenize

DEFAULT_FILENAME = "autocompleter.pkl"
RIGHT_SENT_PAD = "</s>"
LEFT_SENT_PAD = "<s>"


class Autocompleter:
    """Generates text completions using a prefix text and a conditional probability model.

    The autocompleter is based on a ngram language model. The model is constructed
    using up to trigrams and a train corpus is needed to construct the model

    Attributes:
        ngram_counter: NgramCounter used to count the ngrams generated from the train text.

    Typical usage example:

    text_completer = Autocompleter()
    text_completer.fit("train_corpus.json")
    text_completer.generate_completions("How are yo")
    """

    def __init__(self):
        """Initializes ngram_counter to none"""

        self.ngram_counter = None

    def fit(self, json_filename, company_group_id_filter=None, verbose=False):
        """Fits the model with the text in a json file.

        A first step of text pre-processing is performed before model construction. This
        may take some while depending on your processor. Set verbosity to True to have feedback of
        the current construction stage.

        Args:
            json_filename: a json file containing the text used to construct the module.
                (...a description of the json_filename format should also be present here...)
            company_group_id_filter: list of company group ids to consider in the json file. If None,
                then all companies are considered
            verbose: controls the verbosity level (on/off)

        Raises:
            FileNotFoundError: if json_filename does not exist.
        """

        print("Autocompleter fit step 1/7: loading json file...", end='') if verbose else None
        conversations = json.load(open(json_filename))
        operator_messages = util.get_operator_messages(conversations, company_group_id_filter)
        print("done") if verbose else None

        # Sentence tokenize the text
        print("Autocompleter fit step 2/7: sentence tokenize..", end='') if verbose else None
        tokenized_sentences = util.sentence_tokenize(operator_messages)
        print("done") if verbose else None

        # Format sentences, e.g. lower case all text
        print("Autocompleter fit step 3/7: format and clean...", end='') if verbose else None
        tokenized_sentences = [util.format_and_clean(sentence) for sentence in tokenized_sentences]
        print("done") if verbose else None

        # Spell correct the sentences
        print("Autocompleter fit step 4/7: spell correct...", end='') if verbose else None
        tokenized_sentences = [util.spell_correction(sentence) for sentence in tokenized_sentences]
        print("done") if verbose else None

        # Replace named entities with special labels
        print("Autocompleter fit step 5/7: encode named entities, this may take a few minutes...", end='') if verbose else None
        tokenized_sentences = [util.encode_entities(sentence) for sentence in tokenized_sentences]
        print("done") if verbose else None

        # Word tokenize
        print("Autocompleter fit step 6/7: word tokenize...", end='') if verbose else None
        tokenized_words = [tokenize.word_tokenize(sentence) for sentence in tokenized_sentences]
        print("done") if verbose else None

        # Construct ngrams with left and right padding using <s> and </s> correspondingly
        # and construct an n gram counter
        print("Autocompleter fit step 7/7: model construction...", end='') if verbose else None
        trigrams = [nltk.ngrams(sentence, 3, True, True, LEFT_SENT_PAD, RIGHT_SENT_PAD) for sentence in tokenized_words]
        bigrams = [nltk.ngrams(sentence, 2, True, True, LEFT_SENT_PAD, RIGHT_SENT_PAD) for sentence in tokenized_words]
        unigrams = [nltk.ngrams(sentence, 1, True, True, LEFT_SENT_PAD, RIGHT_SENT_PAD) for sentence in tokenized_words]
        self.ngram_counter = counter.NgramCounter(trigrams+bigrams+unigrams)
        print("done") if verbose else None

        print(self.ngram_counter) if verbose else None

    def generate_completions(self, prefix_string, max_sentence_length=6, max_num_of_sentences=2 ):
        """Generate at most max_num_of_sentences sentences of at most max_sentence_length matching a prefix string.

        This method will try to find the most probable sentences (according to the train corpus)
        that starts with the prefix string given. max_sentence_length is just an upper limit, shorter
        sentences could be returned. Same with max_num_of_sentences, it is an upper limit, less sentences could be
        returned.

        Args:
            prefix_string: prefix string used to generate the sentences. Must be a string, e.g. "This is one".
                Can also have an incomplete word at the end: "This is o".
                At least two words must be present so that completions can be inferred from the prefix
            max_sentence_length: the upper limit of the sentence length returned, in words
            max_num_of_sentences: the upper limit of the amount of sentence completions returned

        Returns:
            A list of sentences, e.g., ["This is one sentence", "This is one sentence returned"]
        """

        # Return empty if not initialized
        if self.ngram_counter is None:
            return []

        # Pre-process incoming text: clean & format, spell correct, entity encode, tokenize...
        prefix_string = util.format_and_clean(prefix_string)
        prefix_string = util.spell_correction(prefix_string)
        prefix_string = util.encode_entities(prefix_string)
        tokenized_words = tokenize.word_tokenize(prefix_string)

        # If no tokenized words, the return empty
        if len(tokenized_words) == 0:
            return []

        # Check if last word is in vocabulary, if not use it as the start condition for the next top word search
        start_string = ""
        if not self.in_vocabulary(tokenized_words[-1]):
            start_string = tokenized_words[-1]
            del tokenized_words[-1]

        # Search for the top n first words
        top_predicted_words = self.get_next_word(tokenized_words, top_n=max_num_of_sentences, start_string=start_string)

        # Generate N sentences by using the first N words predicted as prefixes
        # Each sentence returned by get_sentence method is detokenized and true cased
        completions = []
        for word in top_predicted_words:
            sentence_seed = tokenized_words + [word]
            sentence = self.get_sentence(sentence_seed, max_sentence_length)
            sentence = tokenize.treebank.TreebankWordDetokenizer().detokenize(sentence)
            sentence = util.get_true_case(sentence)
            completions.append(sentence)

        return completions

    def in_vocabulary(self, word, count_threshold=50):
        """Returns true if word is in vocabulary more than count_threshold times"""
        if self.ngram_counter is None:
            return False
        return self.ngram_counter[word] > count_threshold

    def get_sentence(self, prefix_string, max_sentence_length=8):
        """Get sentences of max_sentence_length at most, matching a prefix string.

        This method will try to find the most probable sentence (according to the train corpus)
        that starts with the prefix string given. max_sentence_length is just an upper limit, shorter
        sentences can be returned

        Args:
            prefix_string: prefix string used to generate the sentence. Must be a list of words, e.g. ["how", "are"].
                Can also have an incomplete word at the end of the list: ["how", "ar"].
                At least two words must be present so that a sentence can be inferred from the prefix
            max_sentence_length: the upper limit of the sentence length returned, in words

        Returns:
            A sentence is composed as a list of words (strings), e.g., ["This", "is", "a", "sentence"]
        """

        # Return empty if not initialized
        if self.ngram_counter is None:
            return []

        # Check max sentence length precondition
        if len(prefix_string) >= max_sentence_length or len(prefix_string) == 0:
            return prefix_string

        # Copy prefix_string
        prefix_string = list(prefix_string)

        # Construct sentence starting with string_prefix and with max_sentence_length words
        # If sentence terminator is reached or no more words are predicted, the return with current sentence
        sentence = prefix_string
        for i in range(max_sentence_length-len(prefix_string)):
            word = self.get_next_word(sentence, 1)
            if len(word) == 0 or word[0] == RIGHT_SENT_PAD:
                return sentence
            sentence.append(word[0])

        return sentence

    def get_next_word(self, prefix_string, top_n=1, start_string=""):
        """Get next top n most probable words that should follow a string prefix.

        This method will try to find the most probable words (top n according to the train corpus)
        that starts with the prefix string given. start_string is an additional constraint to the word search

        Args:
            prefix_string: prefix string used to search the next words. Must be a list of words, e.g. ["how", "are"].
                Can also have an incomplete word at the end of the list: ["how", "ar"],
                At least two words must be present so that a word can be inferred from the prefix
            top_n: the number of words that should be inferred using the prefix. If less words than top_n are found,
                then just the found words will be returned.
            start_string: additional constraint over the searched words

        Returns:
            A list of words (strings) found
        """

        # Return empty if not initialized
        if self.ngram_counter is None:
            return []

        # Precondition: we need at least two words to generate the next word
        if len(prefix_string) < 1:
            return []

        # Copy prefix string
        prefix_string = list(prefix_string)

        # Get top n words conditioned on last two words that start with start_string
        conditional_word_tuple = tuple(prefix_string[-2:])
        number_of_words = self.ngram_counter.unigrams.N()
        predicted_words = self.ngram_counter[conditional_word_tuple].most_common(number_of_words)
        top_n_words = [word for word, frequency in predicted_words if word.startswith(start_string)][:top_n]
        return top_n_words

    def save(self, filename=DEFAULT_FILENAME):
        """Saves the serialized model to filename."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename=DEFAULT_FILENAME):
        """De-serializes the model from filename."""
        with open(filename, "rb") as f:
            return pickle.load(f)
