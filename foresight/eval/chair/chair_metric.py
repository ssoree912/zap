"""CHAIR (Rohrbach et al., 2018) object-hallucination metric for image captioning.

Ported from https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py
(fetched 2026-07-30), trimmed to CHAIR_s / CHAIR_i only -- the original file
couples hallucination scoring to a pycocoevalcap Bleu/METEOR/CIDEr/SPICE
output format that this port does not need. The synonym list (synonyms.txt,
vendored verbatim from the same repo's data/synonyms.txt) and the
double-word/animal/vehicle special-case tables below are copied as-is;
`caption_to_words`'s tokenize -> singularize -> double-word-merge ->
synonym-filter pipeline runs in the same order as the original so that a
caption is judged hallucinated/not under the same rules the CHAIR paper and
everything that cites CHAIR_s/CHAIR_i numbers uses.

Ground truth objects per image come from two MSCOCO sources, unioned exactly
as the original does: (a) instance segmentation category labels, and (b) any
MSCOCO object mentioned in one of the 5 human reference captions for that
image -- an object doesn't count as "hallucinated" if a human captioner
plausibly saw and named it, even when it's outside the 80 detection
categories.
"""

import json
import os

import nltk

from inflect_en import singularize

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_synonyms():
    with open(os.path.join(_HERE, "synonyms.txt")) as f:
        lines = f.readlines()
    return [line.strip().split(", ") for line in lines]


class CHAIR:
    def __init__(self, imids, instances_paths, captions_paths):
        self.imid_to_objects = {imid: set() for imid in imids}
        self.instances_paths = instances_paths
        self.captions_paths = captions_paths

        synonyms = _load_synonyms()
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]
        self.mscoco_objects = set(self.mscoco_objects)

        coco_double_words = [
            "motor bike", "motor cycle", "air plane", "traffic light", "street light",
            "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
            "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
            "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear", "hair drier",
            "potted plant", "bow tie", "laptop computer", "stove top oven", "hot dog",
            "teddy bear", "home plate", "train track",
        ]
        animal_words = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra", "giraffe", "animal", "cub"]
        vehicle_words = ["jet", "train"]

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict["baby %s" % animal_word] = animal_word
            self.double_word_dict["adult %s" % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict["passenger %s" % vehicle_word] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def caption_to_words(self, caption):
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]

        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i:i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        if ("toilet" in words) and ("seat" in words):
            words = [word for word in words if word != "seat"]

        idxs = [idxs[idx] for idx, word in enumerate(words) if word in self.mscoco_objects]
        words = [word for word in words if word in self.mscoco_objects]
        node_words = [self.inverse_synonym_dict[word] for word in words]
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        id_to_name = {}
        for path in self.instances_paths:
            with open(path) as f:
                data = json.load(f)
            for cat in data["categories"]:
                id_to_name[cat["id"]] = cat["name"]
            for annotation in data["annotations"]:
                imid = annotation["image_id"]
                if imid in self.imid_to_objects:
                    node_word = self.inverse_synonym_dict[id_to_name[annotation["category_id"]]]
                    self.imid_to_objects[imid].add(node_word)

    def get_annotations_from_captions(self):
        for path in self.captions_paths:
            with open(path) as f:
                data = json.load(f)
            for annotation in data["annotations"]:
                imid = annotation["image_id"]
                if imid in self.imid_to_objects:
                    _, node_words, _, _ = self.caption_to_words(annotation["caption"])
                    self.imid_to_objects[imid].update(node_words)

    def get_annotations(self):
        self.get_annotations_from_segments()
        self.get_annotations_from_captions()

    def compute_chair(self, generated_captions):
        """generated_captions: list of {"image_id": int, "caption": str}."""
        num_caps = 0
        num_hallucinated_caps = 0
        hallucinated_word_count = 0
        coco_word_count = 0
        sentences = []

        for cap_eval in generated_captions:
            cap = cap_eval["caption"]
            imid = cap_eval["image_id"]

            words, node_words, idxs, raw_words = self.caption_to_words(cap)
            gt_objects = self.imid_to_objects[imid]

            cap_dict = {
                "image_id": imid,
                "caption": cap,
                "mscoco_hallucinated_words": [],
                "mscoco_gt_words": list(gt_objects),
                "mscoco_generated_words": list(node_words),
                "hallucination_idxs": [],
                "words": raw_words,
            }

            coco_word_count += len(node_words)
            hallucinated = False
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1
                    cap_dict["mscoco_hallucinated_words"].append((word, node_word))
                    cap_dict["hallucination_idxs"].append(idx)
                    hallucinated = True

            num_caps += 1
            if hallucinated:
                num_hallucinated_caps += 1

            cap_dict["chair_s"] = int(hallucinated)
            cap_dict["chair_i"] = (len(cap_dict["mscoco_hallucinated_words"]) / float(len(words))
                                    if len(words) > 0 else 0.0)
            sentences.append(cap_dict)

        chair_s = num_hallucinated_caps / num_caps if num_caps else 0.0
        chair_i = hallucinated_word_count / coco_word_count if coco_word_count else 0.0

        return {
            "chair_s": chair_s,
            "chair_i": chair_i,
            "num_captions": num_caps,
            "num_hallucinated_captions": num_hallucinated_caps,
            "num_mscoco_words": coco_word_count,
            "num_hallucinated_words": hallucinated_word_count,
            "sentences": sentences,
        }
