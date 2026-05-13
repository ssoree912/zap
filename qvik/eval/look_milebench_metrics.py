# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LOOK-M-compatible MileBench scoring helpers.

This module mirrors the local LOOK-M evaluator closely enough to score
generated ``pred.json`` files without importing a specific LOOK-M branch.  It
keeps the same output files expected by LOOK-M experiments: ``eval.json``,
``eval_score.json``, and, for multi-choice tasks, ``pred_with_extracted.json``.
"""

from __future__ import annotations

import re
from typing import Any, Optional


class LookMileBenchEvaluator:
    """Compatibility scorer for MileBench outputs using LOOK-M rules."""

    def __init__(self) -> None:
        self.period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def char(index: int) -> str:
        if index < 26:
            return chr(index + 65)
        if index < 52:
            return "A" + chr(index + 65 - 26)
        return "B" + chr(index + 65 - 52)

    @staticmethod
    def _prediction_index(predictions: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
        index: dict[Any, dict[str, Any]] = {}
        for item in predictions:
            sample_id = item["sample_id"]
            index[sample_id] = item
            index[str(sample_id)] = item
            try:
                index[int(sample_id)] = item
            except Exception:  # noqa: BLE001
                pass
        return index

    def process_punctuation(self, text: str) -> str:
        out_text = text
        for punct in self.punct:
            if (punct + " " in text or " " + punct in text) or re.search(self.comma_strip, text) is not None:
                out_text = out_text.replace(punct, "")
            else:
                out_text = out_text.replace(punct, " ")
        out_text = self.period_strip.sub("", out_text, re.UNICODE)
        return out_text

    def process(self, answer: str) -> str:
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.process_punctuation(answer)
        answer = answer.strip("'")
        answer = answer.strip('"')
        answer = answer.strip().lower()
        return answer

    @staticmethod
    def get_image_quantity_level(sample: dict[str, Any]) -> str:
        image_num = len(sample.get("image", []))
        if image_num < 6:
            return "Few"
        if image_num > 31:
            return "Many"
        return "Medium"

    def _preprocess_option_string(self, option_string: str) -> str:
        processed_option = self.process(option_string)
        special_chars = ["\\", ".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "|", "(", ")"]
        for char in special_chars:
            if char in processed_option:
                processed_option = processed_option.replace(char, "\\" + char)
        return processed_option

    def match_choice(self, text: str, option: dict[str, str]) -> str:
        if text == "":
            return "C"
        try:
            option_str = "|".join([self._preprocess_option_string(f"{k} {v}") for k, v in option.items()])
            option_pattern = rf"({option_str})"
            option_res = re.search(option_pattern, text, re.S)
            if option_res:
                return option_res.group(0)[0].upper()

            option_str = "|".join([self._preprocess_option_string(v).replace(" ", "") for _, v in option.items()])
            option_pattern = rf"({option_str})"
            option_res = re.search(option_pattern, text.replace(" ", ""), re.S)
            if option_res:
                for key, value in option.items():
                    if option_res[0].strip() == self._preprocess_option_string(value).replace(" ", ""):
                        return key.upper()

            if len(text) in [1, 2] and text.upper() in option.keys():
                return text.upper()
        except Exception:  # noqa: BLE001
            return text
        return "".join([char.upper() for char in text if char.upper() in option])

    def process_sample(self, sample: dict[str, Any]) -> None:
        sample["gt_response"] = self.process(str(sample["gt_response"]))
        sample["pred_response"] = self.process(str(sample["pred_response"]))
        for index in range(len(sample["choice_list"])):
            sample["choice_list"][index] = self.process(str(sample["choice_list"][index]))

    def judge_multi_choice(self, sample: dict[str, Any]) -> tuple[int, str]:
        gt_answer = sample["gt_response"]
        pred_answer = sample["pred_response"]
        choice_list = sample["choice_list"]
        if gt_answer not in choice_list:
            raise ValueError(f"Ground truth answer {gt_answer!r} is not present in the choice list")
        option_dict = {self.char(index): choice for index, choice in enumerate(choice_list)}
        selected_answer = self.match_choice(pred_answer, option_dict)
        gt_answer_char = self.char(choice_list.index(sample["gt_response"]))
        return (1, selected_answer) if selected_answer == gt_answer_char else (0, selected_answer)

    def evaluate_multichoice(
        self, predictions: list[dict[str, Any]], core_json: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
        if len(predictions) != len(core_json["data"]):
            raise ValueError(f"There is prediction absent. {len(predictions)}!={len(core_json['data'])}")

        prediction_index = self._prediction_index(predictions)
        for sample in core_json["data"]:
            sample_id = sample["sample_id"]
            if sample_id not in prediction_index:
                raise ValueError(f"Missing prediction for sample_id={sample_id}")
            pred_item = prediction_index[sample_id]
            pred_item["choice_list"] = list(sample["task_instance"]["choice_list"])
            pred_item["image_quantity_level"] = sample.get("image_quantity_level", "Few")
            pred_item["image"] = pred_item.get("image", sample["task_instance"].get("images_path", []))

        correct = 0
        eval_list: list[dict[str, str]] = []
        image_quantity_level_cnt: dict[str, list[float]] = {"Few": [], "Medium": [], "Many": []}
        for sample in predictions:
            self.process_sample(sample)
            score, extracted_answer = self.judge_multi_choice(sample)
            sample["extracted"] = extracted_answer
            sample["result"] = score
            eval_list.append({"id": str(sample["sample_id"]), "score": str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(float(score))

        metrics = {
            "Accuracy": correct / len(predictions),
            "image_quantity_level-Accuracy": {
                key: self._mean(values) if values else 0.0 for key, values in image_quantity_level_cnt.items()
            },
            "image_quantity_level-Result": {
                key: [sum(values), len(values)] for key, values in image_quantity_level_cnt.items()
            },
        }
        return predictions, metrics, eval_list

    @staticmethod
    def _lcs_length(tokens_a: list[str], tokens_b: list[str]) -> int:
        if not tokens_a or not tokens_b:
            return 0
        dp = [0] * (len(tokens_b) + 1)
        for token_a in tokens_a:
            prev = 0
            for j, token_b in enumerate(tokens_b, start=1):
                temp = dp[j]
                if token_a == token_b:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[-1]

    def rouge_l_f1(self, prediction: str, reference: str) -> float:
        pred = self.process(str(prediction))
        ref = self.process(str(reference))
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        lcs = self._lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            return 0.0

        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    @staticmethod
    def _extract_open_ended_gt(sample: dict[str, Any]) -> str:
        for key in ("response", "answer", "target", "label"):
            value = sample.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                return str(value[0]) if value else ""
            return str(value)

        task = sample.get("task_instance")
        if isinstance(task, dict):
            for key in ("response", "answer", "target", "label"):
                value = task.get(key)
                if value is None:
                    continue
                if isinstance(value, list):
                    return str(value[0]) if value else ""
                return str(value)
        return ""

    def evaluate_openended(
        self, predictions: list[dict[str, Any]], core_json: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
        if len(predictions) != len(core_json["data"]):
            raise ValueError(f"There is prediction absent. {len(predictions)}!={len(core_json['data'])}")

        prediction_index = self._prediction_index(predictions)
        for sample in core_json["data"]:
            sample_id = sample["sample_id"]
            if sample_id not in prediction_index:
                raise ValueError(f"Missing prediction for sample_id={sample_id}")

            pred_item = prediction_index[sample_id]
            pred_item["gt_response"] = pred_item.get("gt_response", self._extract_open_ended_gt(sample))
            pred_item["image_quantity_level"] = sample.get("image_quantity_level", "Few")
            pred_item["image"] = pred_item.get("image", sample.get("task_instance", {}).get("images_path", []))

        eval_list: list[dict[str, str]] = []
        image_quantity_level_cnt: dict[str, list[float]] = {"Few": [], "Medium": [], "Many": []}
        scores: list[float] = []

        for sample in predictions:
            score = self.rouge_l_f1(sample.get("pred_response", ""), sample.get("gt_response", ""))
            sample["result"] = score
            scores.append(score)
            eval_list.append({"id": str(sample["sample_id"]), "score": str(round(score, 3))})
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(float(score))

        metrics = {
            "Rouge-L f": self._mean(scores),
            "image_quantity_level-Accuracy": {
                key: self._mean(values) if values else 0.0 for key, values in image_quantity_level_cnt.items()
            },
            "image_quantity_level-Result": {
                key: [sum(values), len(values)] for key, values in image_quantity_level_cnt.items()
            },
        }
        return predictions, metrics, eval_list

    def evaluate_needle(
        self,
        predictions: list[dict[str, Any]],
        core_json: dict[str, Any],
        *,
        needle: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
        if len(predictions) != len(core_json["data"]):
            raise ValueError(f"There is prediction absent. {len(predictions)}!={len(core_json['data'])}")

        prediction_index = self._prediction_index(predictions)
        for sample in core_json["data"]:
            sample_id = sample["sample_id"]
            if sample_id not in prediction_index:
                raise ValueError(f"Missing prediction for sample_id={sample_id}")
            pred_item = prediction_index[sample_id]
            pred_item["image_quantity_level"] = sample.get("image_quantity_level", "Few")
            pred_item["image"] = pred_item.get("image", sample.get("task_instance", {}).get("images_path", []))

        correct = 0
        eval_list: list[dict[str, str]] = []
        image_quantity_level_cnt: dict[str, list[float]] = {"Few": [], "Medium": [], "Many": []}
        for sample in predictions:
            gt_answer = self.process(str(sample.get("gt_response", "")))
            pred_answer = self.process(str(sample.get("pred_response", "")))
            if needle:
                score = 1 if gt_answer in pred_answer.split() else 0
            else:
                score = 1 if gt_answer in pred_answer else 0
            sample["result"] = score
            eval_list.append({"id": str(sample["sample_id"]), "score": str(score)})
            correct += score
            image_quantity_level_cnt[self.get_image_quantity_level(sample)].append(float(score))

        metrics = {
            "Accuracy": correct / len(predictions),
            "image_quantity_level-Accuracy": {
                key: self._mean(values) if values else 0.0 for key, values in image_quantity_level_cnt.items()
            },
            "image_quantity_level-Result": {
                key: [sum(values), len(values)] for key, values in image_quantity_level_cnt.items()
            },
        }
        return predictions, metrics, eval_list

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        core_json: dict[str, Any],
        *,
        dataset_name: Optional[str] = None,
    ) -> tuple[Optional[list[dict[str, Any]]], dict[str, Any], list[dict[str, str]]]:
        if dataset_name and ("NeedleInAHaystack" in dataset_name or "MMCoQA" in dataset_name):
            return self.evaluate_needle(
                predictions,
                core_json,
                needle="NeedleInAHaystack" in dataset_name,
            )

        question_type = str(core_json["meta_data"].get("question_type", "")).lower()
        if question_type == "multi-choice":
            predictions_with_extracted, metrics, eval_list = self.evaluate_multichoice(predictions, core_json)
            return predictions_with_extracted, metrics, eval_list

        if question_type in {"open-ended", "open_ended", "openended"}:
            predictions_with_scores, metrics, eval_list = self.evaluate_openended(predictions, core_json)
            return predictions_with_scores, metrics, eval_list

        raise NotImplementedError(
            "LOOK-compatible scoring is only implemented for multi-choice, open-ended, "
            f"needle, and MMCoQA datasets in this helper, got {question_type!r}"
        )
