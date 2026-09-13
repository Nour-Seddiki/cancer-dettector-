"""CheXpert-style 14-condition labels for IU X-Ray.

PROJECT_PLAN.md Section 4 calls for labelling both the images (stage a, multi-label
classification) and the generated/reference reports (week 6, clinical efficacy) with the
standard CheXpert 14 conditions.

CheXbert is the right tool for that and is what the literature uses, but it needs a
downloaded BERT checkpoint. This module is the dependency-free stand-in:

  * `labels_from_mesh` - uses the MeSH terms that ship *with* IU X-Ray (the plan's
    "labels either shipped with the dataset" option). These are human-assigned, so they
    are the more trustworthy source and are preferred when available.
  * `labels_from_text` - a negation-aware keyword labeller over free report text. This is
    the fallback for studies with no usable MeSH terms, and the only option for
    *generated* reports at evaluation time.

Both return a length-14 vector of {0, 1} (present / not present). Swap in CheXbert later
by replacing `labels_from_text` - see the README section "Upgrading the labeller".
"""

import re

# Canonical CheXpert order - keep this fixed, checkpoints store head weights in it.
CONDITIONS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
NUM_CONDITIONS = len(CONDITIONS)
COND_INDEX = {c: i for i, c in enumerate(CONDITIONS)}
NO_FINDING_IDX = COND_INDEX["No Finding"]

# Keyword patterns per condition, matched against lowercased text.
# A single hit anywhere in a sentence sets the condition for the whole report.
PATTERNS = {
    "Enlarged Cardiomediastinum": [
        r"enlarged cardiomediastinum",
        r"mediastin\w* (?:widen|enlarge)\w*",
        r"widen\w* (?:of the )?mediastin\w*",
        r"cardiomediastin\w* silhouette is enlarged",
    ],
    "Cardiomegaly": [
        r"cardiomegaly",
        r"enlarged (?:cardiac|heart)",
        r"(?:cardiac|heart) (?:size |silhouette )?(?:is |are )?enlarge\w*",
        r"enlargement of the (?:cardiac|heart)",
    ],
    "Lung Opacity": [
        r"opacit\w+", r"opacification", r"airspace disease", r"air space disease",
        r"infiltrat\w+", r"interstitial (?:markings|prominence|disease|pattern)",
        r"reticular", r"densit(?:y|ies)", r"\bhazy\b", r"scarring", r"fibrosis",
        r"emphysema", r"bronchiectasis",
    ],
    "Lung Lesion": [
        r"nodul\w+", r"\bmass(?:es)?\b", r"\blesion\w*", r"granuloma\w*", r"cavit\w+",
        r"carcinoma", r"neoplas\w+", r"tumou?r",
    ],
    "Edema": [
        r"edema", r"oedema", r"vascular congestion", r"pulmonary congestion",
        r"fluid overload", r"\bchf\b", r"congestive heart failure",
    ],
    "Consolidation": [r"consolidat\w+"],
    "Pneumonia": [r"pneumonia", r"infectious process", r"bronchopneumonia"],
    "Atelectasis": [
        r"atelecta\w+", r"collapse(?:d)?\b", r"volume loss", r"hypoinflation",
    ],
    "Pneumothorax": [r"pneumothora\w+"],
    "Pleural Effusion": [
        r"pleural effusion", r"\beffusion\w*", r"pleural fluid", r"hydrothorax",
        r"blunting of the (?:\w+ )?costophrenic",
    ],
    "Pleural Other": [
        r"pleural thickening", r"pleural scarring", r"pleural plaque\w*",
        r"fibrothorax", r"pleural calcification", r"empyema",
    ],
    "Fracture": [r"fractur\w+"],
    "Support Devices": [
        r"catheter\w*", r"\bpicc\b", r"pacemaker", r"\bicd\b", r"defibrillator",
        r"endotracheal tube", r"\bett\b", r"tracheostomy", r"nasogastric",
        r"\bng tube\b", r"chest tube", r"thoracostomy", r"\bstent\w*",
        r"surgical clip\w*", r"sternotomy (?:wire|suture)\w*",
        r"support device\w*", r"valve replacement", r"medical device",
    ],
}
COMPILED = {cond: [re.compile(p) for p in pats] for cond, pats in PATTERNS.items()}

# Phrases that assert "nothing abnormal" - these set No Finding.
NORMAL_PATTERNS = [re.compile(p) for p in [
    r"\bnormal\b",
    r"no acute (?:cardiopulmonary )?(?:abnormalit\w+|finding\w*|disease|process)",
    r"unremarkable",
    r"lungs? (?:are )?clear",
    r"clear lungs?",
    r"without acute",
]]

# Negation / uncertainty cues. If one of these appears within NEG_WINDOW characters
# *before* a condition hit, the hit is discarded - the same windowed-negation heuristic
# the original rule-based CheXpert labeller uses, just much smaller in scope.
NEG_CUES = re.compile(
    r"\b(?:no|not|without|absent|free of|negative for|resolved|rule[sd]? out|"
    r"ruled out|denies|unlikely)\b"
)
NEG_WINDOW = 60

# Conjunctions that close a negation scope: "no effusion but there is opacity".
SCOPE_BREAK = re.compile(r"\b(?:but|however|although|though|except)\b")

# Sentence splitting that tolerates the XXXX anonymisation tokens in IU X-Ray.
_SENT_SPLIT = re.compile(r"(?<=[.;])\s+")


def _is_negated(sentence, match_start):
    """True if a negation cue precedes the match closely enough to scope over it."""
    window = sentence[max(0, match_start - NEG_WINDOW):match_start]
    if not NEG_CUES.search(window):
        return False
    return not SCOPE_BREAK.search(window)


def labels_from_text(text):
    """Negation-aware keyword labelling of a free-text report -> length-14 0/1 list."""
    vec = [0] * NUM_CONDITIONS
    if not text:
        return vec

    text = text.lower()
    for sentence in _SENT_SPLIT.split(text):
        for cond, regexes in COMPILED.items():
            idx = COND_INDEX[cond]
            if vec[idx]:
                continue
            for rx in regexes:
                m = rx.search(sentence)
                if m and not _is_negated(sentence, m.start()):
                    vec[idx] = 1
                    break

    if not any(v for i, v in enumerate(vec) if i != NO_FINDING_IDX):
        # Only call it "No Finding" when the report actually asserts normality. A report
        # that is merely unparseable stays all-zero rather than becoming a false negative.
        if any(rx.search(text) for rx in NORMAL_PATTERNS):
            vec[NO_FINDING_IDX] = 1
    return vec


# IU X-Ray ships MeSH terms per study, e.g. "Cardiomegaly", "Pulmonary Atelectasis",
# "Opacity/lung/base/left", "normal". Map that vocabulary onto the 14 conditions.
MESH_PATTERNS = {
    "Enlarged Cardiomediastinum": [r"mediastin"],
    "Cardiomegaly": [r"cardiomegaly", r"heart.*enlarge", r"hypertrophy.*ventric"],
    "Lung Opacity": [
        r"opacity", r"infiltrate", r"airspace disease", r"emphysema",
        r"pulmonary fibrosis", r"cicatrix", r"interstitial", r"markings",
        r"density", r"bronchiectasis", r"hyperdistention", r"pulmonary disease",
    ],
    "Lung Lesion": [
        r"nodule", r"granuloma", r"\bmass\b", r"carcinoma", r"neoplasm", r"\bcyst",
    ],
    "Edema": [r"pulmonary edema", r"^edema", r"hypertension, pulmonary"],
    "Consolidation": [r"consolidation"],
    "Pneumonia": [r"pneumonia"],
    "Atelectasis": [r"atelectasis", r"hypoinflation", r"collapse"],
    "Pneumothorax": [r"pneumothorax"],
    "Pleural Effusion": [r"pleural effusion", r"hydrothorax"],
    "Pleural Other": [
        r"pleural.*(?:thicken|plaque|calcif)", r"pleural diseases", r"empyema",
    ],
    "Fracture": [r"fracture"],
    "Support Devices": [
        r"catheters", r"pacemaker", r"defibrillator", r"stents", r"prosthesis",
        r"sutures", r"surgical instruments", r"implant", r"device",
    ],
}
MESH_COMPILED = {c: [re.compile(p) for p in pats] for c, pats in MESH_PATTERNS.items()}

# MeSH terms that carry no finding information at all.
MESH_NORMAL = re.compile(r"^\s*normal\s*$")
MESH_UNINFORMATIVE = re.compile(r"no indexing|technical quality|^\s*$")


def labels_from_mesh(mesh_terms):
    """Map IU X-Ray MeSH terms -> length-14 0/1 list, or None if unusable.

    `mesh_terms` is an iterable of raw MeSH strings (major + minor), e.g.
    ["Opacity/lung/base/left", "Calcified Granuloma/lung/upper lobe/right"].
    Returning None means "no signal here" so the caller can fall back to the text
    labeller rather than silently recording an all-negative study.
    """
    terms = [t.strip().lower() for t in (mesh_terms or []) if t and t.strip()]
    terms = [t for t in terms if not MESH_UNINFORMATIVE.search(t)]
    if not terms:
        return None

    vec = [0] * NUM_CONDITIONS
    if all(MESH_NORMAL.match(t) for t in terms):
        vec[NO_FINDING_IDX] = 1
        return vec

    for term in terms:
        if MESH_NORMAL.match(term):
            continue
        for cond, regexes in MESH_COMPILED.items():
            idx = COND_INDEX[cond]
            if vec[idx]:
                continue
            if any(rx.search(term) for rx in regexes):
                vec[idx] = 1

    if not any(v for i, v in enumerate(vec) if i != NO_FINDING_IDX):
        return None  # MeSH present but nothing mapped - let the caller fall back to text
    return vec


def label_study(mesh_terms, report_text, prefer_mesh=True):
    """Best-effort label for one study: MeSH first (human-assigned), text as fallback."""
    if prefer_mesh:
        vec = labels_from_mesh(mesh_terms)
        if vec is not None:
            return vec
    return labels_from_text(report_text)


def positives(vec):
    """Condition names that are positive in a label vector - handy for printing."""
    return [CONDITIONS[i] for i, v in enumerate(vec) if v]


if __name__ == "__main__":
    samples = [
        "The heart size is normal. The lungs are clear. No pleural effusion or pneumothorax.",
        "Mild cardiomegaly. Patchy left basilar opacity concerning for pneumonia. "
        "Small right pleural effusion. No pneumothorax.",
        "Left chest tube in place. Interval decrease in pneumothorax.",
        "No acute cardiopulmonary abnormality.",
    ]
    for s in samples:
        print(f"{s[:58]:60s} -> {positives(labels_from_text(s))}")
    print("mesh ->", positives(labels_from_mesh(["Cardiomegaly", "Opacity/lung/base/left"])))
    print("mesh ->", positives(labels_from_mesh(["normal"])))
