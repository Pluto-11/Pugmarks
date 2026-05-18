"""Compute ExtractionMetrics and ValidationMetrics from a pipeline run.

Ground truth format: list of {surface_form, expected_wikidata_qid, expected_kingdom, page}.
"""
from __future__ import annotations

from pugmark.schemas import (
    Candidate,
    ConfirmedTaxon,
    ExtractionMetrics,
    ValidationMetrics,
)


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_extraction_metrics(
    extracted: list[Candidate], truth: list[dict], chapter_text: str
) -> ExtractionMetrics:
    truth_forms = {t["surface_form"].lower() for t in truth}
    extracted_forms = [c.surface_form.lower() for c in extracted]
    extracted_set = set(extracted_forms)

    correct = truth_forms & extracted_set
    precision = len(correct) / len(extracted_set) if extracted_set else 0.0
    recall = len(correct) / len(truth_forms) if truth_forms else 0.0

    # Hallucination = extracted surface_form does not appear in chapter_text at all
    text_lower = chapter_text.lower()
    hallucinations = sum(1 for f in extracted_forms if f not in text_lower)
    hallucination_rate = hallucinations / len(extracted_forms) if extracted_forms else 0.0

    # Per-type breakdown
    by_type: dict[str, ExtractionMetrics] = {}
    type_names = {t.get("entity_type", "taxa") for t in truth} | {
        c.entity_type for c in extracted
    }
    for tn in type_names:
        t_truth = [t for t in truth if t.get("entity_type", "taxa") == tn]
        t_extracted = [c for c in extracted if c.entity_type == tn]
        if not t_truth and not t_extracted:
            continue
        t_truth_forms = {t["surface_form"].lower() for t in t_truth}
        t_extracted_forms = [c.surface_form.lower() for c in t_extracted]
        t_extracted_set = set(t_extracted_forms)
        t_correct = t_truth_forms & t_extracted_set
        t_precision = len(t_correct) / len(t_extracted_set) if t_extracted_set else 0.0
        t_recall = len(t_correct) / len(t_truth_forms) if t_truth_forms else 0.0
        t_hall = (
            sum(1 for f in t_extracted_forms if f not in text_lower)
            / len(t_extracted_forms)
            if t_extracted_forms
            else 0.0
        )
        by_type[tn] = ExtractionMetrics(
            precision=t_precision,
            recall=t_recall,
            f1=_f1(t_precision, t_recall),
            hallucination_rate=t_hall,
            by_type=None,  # leaf
        )

    return ExtractionMetrics(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        hallucination_rate=hallucination_rate,
        by_type=by_type or None,
    )


def compute_validation_metrics(
    confirmed: list[ConfirmedTaxon], unresolved: list[Candidate], truth: list[dict]
) -> ValidationMetrics:
    truth_forms_to_qid = {t["surface_form"].lower(): t["expected_wikidata_qid"] for t in truth}

    correct = 0
    total_should_resolve = 0
    confusion: dict[str, dict[str, int]] = {}

    for taxon in confirmed:
        for cand in taxon.source_candidates:
            sf = cand.surface_form.lower()
            if sf in truth_forms_to_qid:
                total_should_resolve += 1
                expected = truth_forms_to_qid[sf]
                actual = taxon.wikidata_qid
                confusion.setdefault(expected, {}).setdefault(actual, 0)
                confusion[expected][actual] += 1
                if expected == actual:
                    correct += 1

    qid_accuracy = correct / total_should_resolve if total_should_resolve else 0.0
    unresolved_should_have_resolved = sum(
        1 for c in unresolved if c.surface_form.lower() in truth_forms_to_qid
    )
    total_truth = len(truth)
    unresolved_rate = unresolved_should_have_resolved / total_truth if total_truth else 0.0

    return ValidationMetrics(
        qid_accuracy=qid_accuracy,
        confusion_matrix=confusion,
        unresolved_rate=unresolved_rate,
    )
