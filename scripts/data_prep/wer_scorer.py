"""
Word/Character Error Rate scorer, edit-distance based, with punctuation stripped from
both sides before comparison -- our training targets (from VoxForge) never contain
punctuation, but the model's pretrained decoder may still produce some, and a
literal-string WER (like nemo's own wer.py, confirmed to have zero case/punctuation
normalization) would penalize that formatting difference as if it were a real
transcription error.

char_level=True switches from word-level (whitespace .split()) to character-level
comparison -- required for CJK languages, which don't use spaces between words or
characters at all. Confirmed directly: word-level scoring on Chinese text splits an
entire un-spaced sentence into a single "word", so ANY single wrong character scores
the WHOLE utterance as 100% wrong (WER=1.0 for a sentence differing by exactly one
character out of sixteen, verified concretely) -- not a training/model problem, a
measurement artifact from reusing scoring logic built and validated for space-
delimited European languages. Defaults to False (unchanged, word-level behavior) so
every existing European-language evaluation is completely unaffected.
"""
import re


def normalize(text, char_level=False):
    text = text.lower()
    # ASCII/Latin punctuation (original, for VoxForge/European languages)
    text = re.sub(r"[.,!?;:¿¡\"'`\-]", " ", text)
    if char_level:
        # Full-width CJK punctuation -- confirmed via AliMeeting/MagicData corpus scans
        # to be the dominant punctuation convention in real Chinese transcripts, and
        # NOT covered by the ASCII pattern above (different Unicode code points
        # entirely, e.g. full-width "，" U+FF0C vs ASCII "," U+002C).
        text = re.sub(r"[。，？！；：""''、…—]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_error_rate(reference, hypothesis, char_level=False):
    """Returns (wer, n_sub, n_del, n_ins, n_ref_words) for one reference/hypothesis
    pair. Despite the name, computes CER when char_level=True -- caller is
    responsible for interpreting/labeling the result appropriately."""
    if char_level:
        # Split into individual characters, not whitespace-delimited words -- CJK
        # text has no meaningful word-space to split on at all.
        ref_words = list(normalize(reference, char_level=True).replace(" ", ""))
        hyp_words = list(normalize(hypothesis, char_level=True).replace(" ", ""))
    else:
        ref_words = normalize(reference).split()
        hyp_words = normalize(hypothesis).split()

    n, m = len(ref_words), len(hyp_words)
    # Standard Levenshtein DP over words, tracking operation counts.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to classify each edit as substitution / deletion / insertion.
    i, j = n, m
    n_sub = n_del = n_ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            n_sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            n_del += 1
            i -= 1
        else:
            n_ins += 1
            j -= 1

    n_ref = len(ref_words)
    wer = (n_sub + n_del + n_ins) / n_ref if n_ref > 0 else (0.0 if n_ins == 0 else float("inf"))
    return wer, n_sub, n_del, n_ins, n_ref
