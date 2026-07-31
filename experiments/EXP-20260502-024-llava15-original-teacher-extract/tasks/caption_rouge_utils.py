# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the official COCO-caption ROUGE-L metric without Java scorers."""

from pycocoevalcap.rouge.rouge import Rouge
from nltk.tokenize import TreebankWordTokenizer

_PUNCTUATION = {
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
}
_TOKENIZER = TreebankWordTokenizer()


def _tokenize(caption):
    tokens = _TOKENIZER.tokenize(str(caption).replace("\n", " ").lower())
    return " ".join(token for token in tokens if token not in _PUNCTUATION)


def _rouge_l(results, args=None):
    del args
    ground_truths = {}
    predictions = {}
    for index, result in enumerate(results):
        ground_truths[index] = [_tokenize(answer) for answer in result["answer"]]
        predictions[index] = [_tokenize(result["pred"])]

    score, _ = Rouge().compute_score(ground_truths, predictions)
    return score


def coco_rougel(results, args=None):
    return _rouge_l(results, args=args)


def nocaps_rougel(results, args=None):
    return _rouge_l(results, args=args)


def textcaps_rougel(results, args=None):
    return _rouge_l(results, args=args)


__all__ = ["coco_rougel", "nocaps_rougel", "textcaps_rougel"]
