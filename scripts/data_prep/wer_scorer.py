"""
Word Error Rate scorer, edit-distance based, with punctuation stripped from both
sides before comparison -- our training targets (from VoxForge) never contain
punctuation, but the model's pretrained decoder may still produce some, and a
literal-string WER (like nemo's own wer.py, confirmed to have zero case/punctuation
normalization) would penalize that formatting difference as if it were a real
transcription error.
"""
import re


def normalize(text):
    text = text.lower()
    text = re.sub(r"[.,!?;:¿¡\"'`\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_error_rate(reference, hypothesis):
    """Returns (wer, n_sub, n_del, n_ins, n_ref_words) for one reference/hypothesis pair."""
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
