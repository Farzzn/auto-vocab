"""Tokenization, stopword filtering, POS-aware lemmatization, and rarity scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from wordfreq import zipf_frequency

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")

_EXTRA_STOPWORDS = {
    "yeah",
    "yes",
    "no",
    "nope",
    "nah",
    "okay",
    "ok",
    "hey",
    "hi",
    "hello",
    "bye",
    "oh",
    "uh",
    "um",
    "ah",
    "er",
    "hmm",
    "huh",
    "wow",
    "gonna",
    "wanna",
    "gotta",
    "kinda",
    "sorta",
    "lemme",
    "gimme",
    "aint",
    "ain't",
    "cant",
    "can't",
    "couldnt",
    "couldn't",
    "didnt",
    "didn't",
    "doesnt",
    "doesn't",
    "dont",
    "don't",
    "hadnt",
    "hadn't",
    "hasnt",
    "hasn't",
    "havent",
    "haven't",
    "isnt",
    "isn't",
    "mightnt",
    "mightn't",
    "mustnt",
    "mustn't",
    "neednt",
    "needn't",
    "shant",
    "shan't",
    "shouldnt",
    "shouldn't",
    "wasnt",
    "wasn't",
    "werent",
    "weren't",
    "wont",
    "won't",
    "wouldnt",
    "wouldn't",
    "im",
    "i'm",
    "ive",
    "i've",
    "id",
    "i'd",
    "ill",
    "i'll",
    "youre",
    "you're",
    "youve",
    "you've",
    "youd",
    "you'd",
    "youll",
    "you'll",
    "hes",
    "he's",
    "shes",
    "she's",
    "its",
    "it's",
    "were",
    "we're",
    "weve",
    "we've",
    "wed",
    "we'd",
    "well",
    "we'll",
    "theyre",
    "they're",
    "theyve",
    "they've",
    "theyd",
    "they'd",
    "theyll",
    "they'll",
    "thats",
    "that's",
    "whats",
    "what's",
    "heres",
    "here's",
    "theres",
    "there's",
    "wheres",
    "where's",
    "whos",
    "who's",
    "hows",
    "how's",
    "lets",
    "let's",
    "could",
    "would",
    "should",
    "one",
    "two",
    "get",
    "got",
    "like",
    "know",
    "just",
    "really",
    "right",
    "well",
    "see",
    "come",
    "back",
    "make",
    "going",
    "want",
    "think",
    "look",
    "good",
    "time",
    "say",
    "said",
    "tell",
    "told",
}


@dataclass(frozen=True)
class CandidateWord:
    """Represents a filtered vocabulary candidate extracted from a subtitle."""

    surface: str
    lemma: str
    pos_hint: str  # "noun" | "verb" | "adj" | "adv" | ""
    zipf: float


def _penn_to_wordnet(tag: str) -> tuple[str, str]:
    """Map Penn Treebank POS tag to (wordnet_pos, human_pos_hint)."""
    if tag.startswith("J"):
        return wordnet.ADJ, "adj"
    if tag.startswith("V"):
        return wordnet.VERB, "verb"
    if tag.startswith("N"):
        return wordnet.NOUN, "noun"
    if tag.startswith("R"):
        return wordnet.ADV, "adv"
    return wordnet.NOUN, ""


class WordAnalyzer:
    """Extracts, lemmatizes, and frequency-filters vocabulary words from subtitle text."""

    def __init__(self) -> None:
        self._lemmatizer = WordNetLemmatizer()
        try:
            base_sw = set(stopwords.words("english"))
        except LookupError:
            base_sw = set()
        self._stopwords = {w.lower() for w in base_sw} | _EXTRA_STOPWORDS

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in _TOKEN_RE.findall(text):
            norm = raw.replace("’", "'")
            lower = norm.lower()
            if lower in self._stopwords:
                continue
            # Strip possessive 's / '
            if lower.endswith("'s"):
                norm = norm[:-2]
                lower = lower[:-2]
            elif lower.endswith("'"):
                norm = norm[:-1]
                lower = lower[:-1]
            # Skip contractions with internal apostrophe if still present
            if "'" in lower:
                continue
            if len(lower) < 3 or not lower.isalpha():
                continue
            if lower in self._stopwords:
                continue
            tokens.append(norm)
        return tokens

    def extract_all_candidates(self, text: str) -> list[CandidateWord]:
        """Extract all non-stopword candidate words with lemma and Zipf frequency."""
        if not text or not text.strip():
            return []

        tokens = self._tokenize(text)
        if not tokens:
            return []

        try:
            tagged = nltk.pos_tag(tokens)
        except LookupError:
            tagged = [(tok, "NN") for tok in tokens]

        seen_lemmas: set[str] = set()
        candidates: list[CandidateWord] = []

        for surface, tag in tagged:
            lower_surface = surface.lower()
            wn_pos, pos_hint = _penn_to_wordnet(tag)
            lemma = self._lemmatizer.lemmatize(lower_surface, pos=wn_pos)
            if lemma == lower_surface and wn_pos == wordnet.NOUN and not wordnet.synsets(lower_surface, pos=wordnet.NOUN):
                v_lemma = self._lemmatizer.lemmatize(lower_surface, pos=wordnet.VERB)
                if v_lemma != lower_surface and wordnet.synsets(v_lemma, pos=wordnet.VERB):
                    lemma = v_lemma
                    pos_hint = "verb"

            if len(lemma) < 3 or lemma in self._stopwords:
                continue
            if lemma in seen_lemmas:
                continue
            seen_lemmas.add(lemma)

            freq_surface = zipf_frequency(lower_surface, "en")
            freq_lemma = zipf_frequency(lemma, "en")
            zipf = max(freq_surface, freq_lemma)

            candidates.append(
                CandidateWord(
                    surface=lower_surface,
                    lemma=lemma,
                    pos_hint=pos_hint,
                    zipf=round(zipf, 2),
                )
            )

        return candidates

    @staticmethod
    def filter_candidates(
        candidates: Iterable[CandidateWord], zipf_threshold: float = 4.8
    ) -> list[CandidateWord]:
        """Filter candidates keeping only words rarer than zipf_threshold."""
        return [c for c in candidates if c.zipf < zipf_threshold]

    def analyze(self, text: str, zipf_threshold: float = 4.8) -> list[CandidateWord]:
        """Extract and filter candidate words below zipf_threshold."""
        return self.filter_candidates(self.extract_all_candidates(text), zipf_threshold)
