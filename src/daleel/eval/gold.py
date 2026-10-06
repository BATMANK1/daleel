"""The gold question set, the checks it passes, and how it was frozen.

Retrieval and answers are measured against these questions, so a mistake in
them corrupts every table that follows. The set was drafted in
eval/gold_draft.jsonl, and once every answer had been checked by hand against
its page it was frozen as eval/gold_v1.jsonl (freeze), which never changes: a
mistake found later goes into a gold_v2.jsonl, reported beside the first.

The set has a fixed composition, COMPOSITION: 80 questions of eight types,
from single clauses and numbers to questions the corpus cannot answer. Each
question names the pages that answer it, by the PDF's own page numbers from
1, and a topic group: questions that ask the same thing in other words, or in
English, share one, and a group is never divided between the questions used
for development and the 20 held out for the end (choose_final).

What a correct answer does follows from whether the corpus can answer the
question at all. Most are answered and cited. Where two documents disagree,
both are shown. Where they cover different cases, the cases are told apart.
Where the corpus holds only part of the answer, that part is given and what is
missing is named, without inventing it. Only a question about something the
corpus never mentions is refused.

Each answering page carries quotes: the words of the page that hold the
answer, a string each, or a list of strings that must stand together, as the
cells of one table row or the title and date of one calendar card do. Whether
a retrieved chunk holds the evidence is decided by finding the quotes in it,
not by naming chunks, so the frozen set outlives any change to how the corpus
is chunked. Every quote must be found in its page's extracted text, once both
are normalized for comparison.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from daleel.normalize.arabic import for_comparison

GOLD_V1 = Path("eval/gold_v1.jsonl")
# What a frozen question's status says.
FROZEN = "gold_v1"

# Every document a question may cite, and its number of pages, as
# data/README.md lists them.
CORPUS = {
    "academic_weeks_1448": 3,
    "guidance_manual": 19,
    "library_services_2024_2025": 16,
    "organizational_regulations": 58,
    "orientation_1446": 13,
    "student_charter": 8,
    "student_conduct_code": 12,
    "student_guide_2025": 86,
    "student_portal_guide": 42,
}

# How many questions of each type the set holds.
COMPOSITION = {
    "single_clause": 25,
    "numeric": 15,
    "multi_clause": 10,
    "table_lookup": 8,
    "cross_document": 6,
    "precedence": 4,
    "out_of_scope": 8,
    "english_query": 4,
}

# What a correct answer does, by how far the corpus answers the question.
BEHAVIOR = {
    "supported": "answer",
    "document_conflict": "show_both_sources",
    "scope_comparison": "tell_the_cases_apart",
    "unsupported_detail": "answer_and_name_the_gap",
    "needs_course_context": "answer_and_name_the_gap",
    "outside_corpus": "refuse",
}
ANSWERABLE = frozenset({"supported", "document_conflict", "scope_comparison"})

# A reference either holds the answer or only touches the topic, as for a
# question the corpus cannot answer.
ROLES = ("answer_evidence", "related_only")
SPLITS = ("dev", "final")
FINAL_SIZE = 20
# The held-out questions are chosen from this many orders of the topic groups,
# shuffled from this seed, so the choice can be made again and checked.
SPLIT_SEED = 1448
SPLIT_ORDERS = 20_000

FIELDS = frozenset(
    {
        "qid",
        "lang",
        "type",
        "question",
        "source_type",
        "raw_question_numbers",
        "original_questions",
        "context_added",
        "topic_group",
        "answerable",
        "answerability_status",
        "expected_behavior",
        "source_refs",
        "subtype",
        "status",
        "human_reviewed",
        "split",
        "answer",
        "answer_numeric",
        "notes",
    }
)
REF_FIELDS = frozenset({"doc_id", "pdf_page", "section", "role", "verification", "quotes"})
REQUIRED = frozenset(
    {
        "qid",
        "lang",
        "type",
        "question",
        "topic_group",
        "answerable",
        "answerability_status",
        "source_refs",
        "human_reviewed",
        "split",
    }
)
QID = re.compile(r"g\d{3}")


def load_gold(path: Path) -> list[dict]:
    """The questions in a gold JSONL file, in file order."""
    questions = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}, line {number}: {error}") from error
    return questions


def behavior(question: dict) -> str:
    """What a correct answer to the question does."""
    return BEHAVIOR[question["answerability_status"]]


def composition(questions: Sequence[dict]) -> Counter[str]:
    """How many questions there are of each type."""
    return Counter(question.get("type") for question in questions)


def fragments(quotes: Sequence[str | Sequence[str]]) -> list[str]:
    """Every string in a reference's quotes, groups opened up."""
    return [text for quote in quotes for text in ([quote] if isinstance(quote, str) else quote)]


def _quote_problems(qid: str, ref: dict) -> list[str]:
    quotes = ref["quotes"]
    where = f"{qid}, {ref.get('doc_id')} page {ref.get('pdf_page')}"
    if ref.get("role") != "answer_evidence":
        return [f"{where}: quotes on a reference that does not hold the answer"]
    if not isinstance(quotes, list) or not quotes:
        return [f"{where}: quotes is not a list of quotes"]
    for quote in quotes:
        group = [quote] if isinstance(quote, str) else quote
        if not isinstance(group, list) or not group:
            return [f"{where}: a quote is neither a string nor a list of strings"]
        if not all(isinstance(text, str) and for_comparison(text) for text in group):
            return [f"{where}: a quote is empty"]
    return []


def _ref_problems(qid: str, ref: object) -> list[str]:
    if not isinstance(ref, dict):
        return [f"{qid}: a source reference is not an object"]
    found = []
    unknown = sorted(ref.keys() - REF_FIELDS)
    if unknown:
        found.append(f"{qid}: a reference has unknown field {', '.join(unknown)}")
    doc, page = ref.get("doc_id"), ref.get("pdf_page")
    if doc not in CORPUS:
        found.append(f"{qid}: cites {doc!r}, which is not in the corpus")
    elif not isinstance(page, int) or not 1 <= page <= CORPUS[doc]:
        found.append(f"{qid}: cites page {page!r} of {doc}, which has {CORPUS[doc]} pages")
    if ref.get("role") not in ROLES:
        found.append(f"{qid}: a reference's role is {ref.get('role')!r}")
    if "quotes" in ref:
        found += _quote_problems(qid, ref)
    return found


def _answer_problems(qid: str, question: dict, evidence: list[dict]) -> list[str]:
    answer = question.get("answer")
    if answer is None:
        if question["human_reviewed"] is True:
            return [f"{qid}: checked by hand, but there is no answer"]
        return []
    found = []
    if not isinstance(answer, str) or not answer.strip():
        found.append(f"{qid}: an empty answer")
    numeric = question.get("answer_numeric")
    if numeric is not None and (isinstance(numeric, bool) or not isinstance(numeric, int | float)):
        found.append(f"{qid}: answer_numeric is not a number")
    unquoted = [ref for ref in evidence if not ref.get("quotes")]
    if unquoted:
        pages = ", ".join(f"{ref.get('doc_id')} page {ref.get('pdf_page')}" for ref in unquoted)
        found.append(f"{qid}: answered, but no quote from {pages}")
    return found


def question_problems(question: dict) -> list[str]:
    """Everything wrong with one question, or nothing."""
    qid = question.get("qid", "?")
    missing = sorted(REQUIRED - question.keys())
    unknown = sorted(question.keys() - FIELDS)
    found = [f"{qid}: missing {', '.join(missing)}"] if missing else []
    if unknown:
        found.append(f"{qid}: unknown field {', '.join(unknown)}")
    if missing:
        return found

    kind, status = question["type"], question["answerability_status"]
    if not isinstance(qid, str) or not QID.fullmatch(qid):
        found.append(f"{qid}: an id is g and three digits")
    if question["lang"] not in ("ar", "en"):
        found.append(f"{qid}: language {question['lang']!r}")
    if kind not in COMPOSITION:
        found.append(f"{qid}: unknown type {kind!r}")
    if not str(question["question"]).strip():
        found.append(f"{qid}: no question")
    if not str(question["topic_group"]).strip():
        found.append(f"{qid}: no topic group")
    if status not in BEHAVIOR:
        found.append(f"{qid}: unknown answerability {status!r}")
        return found
    if not isinstance(question["human_reviewed"], bool):
        found.append(f"{qid}: human_reviewed is not true or false")
    if question["split"] not in (None, *SPLITS):
        found.append(f"{qid}: split {question['split']!r}")

    answerable = status in ANSWERABLE
    if question["answerable"] is not answerable:
        found.append(f"{qid}: answerable is {question['answerable']}, but its status is {status}")
    conflict = status in ("document_conflict", "scope_comparison")
    if (kind == "out_of_scope") == answerable or (kind == "precedence") != conflict:
        found.append(f"{qid}: type {kind} does not fit status {status}")
    if kind == "english_query" and question["lang"] != "en":
        found.append(f"{qid}: an English query in {question['lang']!r}")

    refs = question["source_refs"]
    if not isinstance(refs, list):
        return [*found, f"{qid}: source_refs is not a list"]
    for ref in refs:
        found += _ref_problems(qid, ref)
    evidence = [
        ref for ref in refs if isinstance(ref, dict) and ref.get("role") == "answer_evidence"
    ]
    if answerable and not evidence:
        found.append(f"{qid}: answerable, but no reference holds the answer")
    if not answerable and evidence:
        found.append(f"{qid}: not answerable, but a reference is marked as holding the answer")
    if status != "outside_corpus" and not refs:
        found.append(f"{qid}: no source reference")
    if kind == "cross_document" and len({ref.get("doc_id") for ref in evidence}) < 2:
        found.append(f"{qid}: a cross-document question answered from one document")
    return found + _answer_problems(qid, question, evidence)


def split_problems(questions: Sequence[dict]) -> list[str]:
    """Problems with how the questions are divided between development and the end."""
    splits = [question.get("split") for question in questions]
    if all(split is None for split in splits):
        return []
    found = []
    if None in splits:
        found.append(f"{splits.count(None)} questions have no split")
    final = splits.count("final")
    if final != FINAL_SIZE:
        found.append(f"{final} questions are held out, not {FINAL_SIZE}")
    groups: dict[str, set] = {}
    for question in questions:
        groups.setdefault(question.get("topic_group"), set()).add(question.get("split"))
    for group, used in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(used) > 1:
            found.append(
                f"topic group {group} is divided between {', '.join(sorted(map(str, used)))}"
            )
    return found


def choose_final(
    questions: Sequence[dict], seed: int = SPLIT_SEED, orders: int = SPLIT_ORDERS
) -> frozenset[str]:
    """The ids of the FINAL_SIZE questions held out for the end, in whole topic groups.

    The held-out questions should ask what the rest ask, so each type should
    be held out in its share of the set: a quarter of each, where whole groups
    allow. The groups are put in `orders` seeded random orders, and each order
    is taken group by group while the next group fits; of the orders that fill
    exactly FINAL_SIZE, the one whose types are nearest their shares wins, the
    first of them on a tie.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for question in questions:
        groups[question["topic_group"]].append(question)
    share = {
        kind: FINAL_SIZE * count / len(questions) for kind, count in composition(questions).items()
    }
    names = sorted(groups)
    shuffle = random.Random(seed).shuffle
    best: list[str] = []
    best_distance = float("inf")
    for _ in range(orders):
        shuffle(names)
        chosen, size = [], 0
        for name in names:
            if size + len(groups[name]) <= FINAL_SIZE:
                chosen.append(name)
                size += len(groups[name])
        if size != FINAL_SIZE:
            continue
        held = Counter(question["type"] for name in chosen for question in groups[name])
        distance = sum(abs(held[kind] - target) for kind, target in share.items())
        if distance < best_distance:
            best, best_distance = list(chosen), distance
    if not best:
        raise ValueError(f"no order of the topic groups holds out exactly {FINAL_SIZE}")
    return frozenset(question["qid"] for name in best for question in groups[name])


def freeze(questions: Sequence[dict]) -> list[dict]:
    """The set as frozen: every question checked by hand, each given its split.

    Refuses a set with any problem, or with a question not yet checked by hand
    against its page.
    """
    found = problems(questions)
    unchecked = [
        question["qid"] for question in questions if question.get("human_reviewed") is not True
    ]
    if unchecked:
        found.append(f"not checked by hand: {', '.join(unchecked)}")
    if found:
        raise ValueError("cannot freeze: " + "; ".join(found))
    final = choose_final(questions)
    return [
        {**question, "status": FROZEN, "split": "final" if question["qid"] in final else "dev"}
        for question in questions
    ]


def problems(questions: Sequence[dict]) -> list[str]:
    """Everything wrong with a gold set, or nothing: each question, the ids,
    the composition, and the split."""
    found = []
    for question in questions:
        found += question_problems(question)
    repeated = sorted(
        qid for qid, count in Counter(q.get("qid") for q in questions).items() if count > 1
    )
    if repeated:
        found.append(f"repeated ids: {', '.join(map(str, repeated))}")
    counts = composition(questions)
    for kind, expected in COMPOSITION.items():
        if counts[kind] != expected:
            found.append(f"{counts[kind]} {kind} questions, not {expected}")
    return found + split_problems(questions)


def evidence_problems(questions: Sequence[dict], pages: Mapping[tuple[str, int], str]) -> list[str]:
    """Every quote not found in its page's text, both normalized for comparison.

    `pages` maps a document and a PDF page number to the page's extracted text.
    """
    found = []
    normalized: dict[tuple[str, int], str] = {}
    for question in questions:
        for ref in question.get("source_refs", []):
            key = (ref.get("doc_id"), ref.get("pdf_page"))
            for text in fragments(ref.get("quotes", [])):
                if key not in pages:
                    found.append(f"{question['qid']}: no text for {key[0]} page {key[1]}")
                    break
                if key not in normalized:
                    normalized[key] = for_comparison(pages[key])
                if for_comparison(text) not in normalized[key]:
                    found.append(
                        f"{question['qid']}: {key[0]} page {key[1]} does not hold {text!r}"
                    )
    return found
