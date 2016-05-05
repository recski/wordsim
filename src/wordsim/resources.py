import string
import re
import math
import nltk


class Resources(object):

    @staticmethod
    def set_config(conf):
        Resources.conf = conf

    @staticmethod
    def ensure_nltk_packages():
        for package in ('stopwords', 'punkt', 'wordnet'):
            nltk.download(package, quiet=True)

    """ Thresholds """
    adverb_threshold = math.log(500000)

    punctuation = set(string.punctuation)
    punct_re = re.compile("\W+")
    num_re = re.compile(r'^([0-9][0-9.,]*)([mMkK]?)$', re.UNICODE)
    question_starters = set([
        'is', 'does', 'do', 'what', 'where', 'how', 'why',
    ])
    pronouns = {
        'me': 'i', 'my': 'i',
        'your': 'you',
        'him': 'he', 'his': 'he',
        'her': 'she',
        'us': 'we', 'our': 'we',
        'them': 'they', 'their': 'they',
    }
    written_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    stopwords = set(
        nltk.corpus.stopwords.words('english')) - set(pronouns.iterkeys())
    twitter_cache = {}
    _global_freqs = None
    _adverb_cache = {}

    @staticmethod
    def is_pronoun_equivalent(word1, word2):
        l1 = word1.lower()
        l2 = word2.lower()
        if l1 in Resources.pronouns and l2 in Resources.pronouns:
            return Resources.pronouns[l1] == Resources.pronouns[l2]
        return False

    @staticmethod
    def get_global_freq(lookup):
        if not Resources._global_freqs:
            Resources._global_freqs = {}
            with open(Resources.conf.get('global', 'freqs')) as f:
                for l in f:
                    try:
                        fd = l.decode('utf8').strip().split(' ')
                        word = fd[1]
                        logfreq = math.log(int(fd[0]) + 2)
                        Resources._global_freqs[word] = logfreq
                    except (ValueError, IndexError):
                        continue
        return Resources._global_freqs.get(lookup, 2)

    @staticmethod
    def is_frequent_adverb(word, pos):
        if word not in Resources._adverb_cache:
            ans = (
                pos is not None and pos[:2] == 'RB' and
                Resources.get_global_freq(word) > Resources.adverb_threshold)
            Resources._adverb_cache[word] = ans
        return Resources._adverb_cache[word]

    @staticmethod
    def is_num_equivalent(word1, word2):
        num1 = Resources.to_num(word1)
        num2 = Resources.to_num(word2)
        if num1 and num2:
            return num1 == num2
        return False

    @staticmethod
    def to_num(word):
        if word in Resources.written_numbers:
            return Resources.written_numbers[word]
        m = Resources.num_re.match(word)
        if not m:
            return False
        num = float(m.group(1).replace(',', ''))
        if m.group(2):
            c = m.group(2).lower()
            if c == 'k':
                num *= 1000
            else:
                num *= 1000000
        return num

    @staticmethod
    def twitter_candidates(word, model):
        if model not in Resources.twitter_cache:
            Resources.twitter_cache[model] = {}
        if word not in Resources.twitter_cache[model]:
            # adding word as hashtag
            candidates = set(['#' + word])
            candidates |= set(Resources.norvig_spellchecker(word))
            candidates.add(Resources.trim_dup_letters(word))
            for a, b in Resources.part_of_vocab(word, model):
                candidates.add(a)
                candidates.add(b)
            Resources.twitter_cache[model][word] = set(
                filter(lambda x: x in model, candidates))
        return Resources.twitter_cache[model][word]

    @staticmethod
    def trim_dup_letters(word):
        new_w = word[0]
        for c in word:
            if not new_w[-1] == c:
                new_w += c
        return new_w

    @staticmethod
    def norvig_spellchecker(word, dist=2):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
        inserts = [a + c + b for a, b in splits for c in alphabet]
        return set(deletes + transposes + inserts + replaces)

    @staticmethod
    def part_of_vocab(word, dictionary):
        if len(word) < 5:
            return []
        splits = [(word[:i], word[i:]) for i in range(3, len(word) - 2)]
        parts = []
        for a, b in splits:
            if a in dictionary and b in dictionary:
                parts.append((a, b))
        return parts
