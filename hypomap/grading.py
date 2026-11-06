"""A small finite checker with an explicit boundary around unscored science.

The checker certifies only reviewed evidence availability/selection constraints
and structural patch safety. It does not certify free-text scientific claims.
"""

from __future__ import annotations

from copy import deepcopy

from .tasks import ASPECT_TYPES, OPERATIONS, OUTPUT_FIELDS, object_ids, stable_hash

GRADER_ID = "scoped-evidence-availability"
GRADER_REVISION = "1"
SCORED_FIELDS = ["aspect_assessments.target_ids", "aspect_assessments.evidence_ids", "aspect_assessments.availability"]
UNSCORED_FIELDS = ["aspect_assessments.assessment", "aspect_assessments.scope", "aspect_assessments.limitations", "aspect_assessments.rationale", "findings", "proposed_updates.scientific_meaning", "verification_needs", "preserved_ids.scientific_adequacy", "information_requests", "no_change_reason"]


def availability_scoring_contract(task: dict) -> dict:
    """Draft a finite contract from declared targets, requiring separate review.

    The public observation records explicitly declare their target aspects.
    This contract tests use of that declared inventory, not independently
    inferred causal relevance, scientific sufficiency, or result direction.
    """
    packet = task["public_input"]
    conditions = []
    for aspect in packet["current_map"]["nodes"]:
        if aspect.get("type") not in ASPECT_TYPES:
            continue
        relevant = [obs["id"] for obs in packet["visible_observations"] if aspect["id"] in obs.get("target_aspect_ids", [])]
        conditions.append({
            "target_id": aspect["id"],
            "availability": "relevant_visible" if relevant else "no_relevant_visible",
            "admissible_evidence_ids": relevant,
            "sufficient_evidence_sets": [[identifier] for identifier in relevant] if relevant else [[]],
            "required_evidence_ids": [], "allow_additional_relevant_evidence": True,
            "certified_claim": "Whether explicitly target-linked observations are present, not whether the scientific claim is established.",
        })
    return {
        "grader_id": GRADER_ID, "grader_revision": GRADER_REVISION,
        "scored_fields": list(SCORED_FIELDS), "unscored_fields": list(UNSCORED_FIELDS),
        "sufficient_evidence_conditions": conditions, "invalid_answer_tests": [], "valid_alternative_tests": [],
    }


def validate_answer(answer: dict, public_input: dict) -> list[str]:
    errors = []
    if not isinstance(answer, dict):
        return ["answer_not_object"]
    if set(answer) != set(OUTPUT_FIELDS):
        errors.append("output_fields_mismatch")
    for field in OUTPUT_FIELDS[:-1]:
        if not isinstance(answer.get(field), list):
            errors.append(f"output_collection_not_list:{field}")
    if answer.get("no_change_reason") is not None and not isinstance(answer["no_change_reason"], str):
        errors.append("invalid_no_change_reason")
    if errors:
        return errors
    known = object_ids(public_input)
    proposed_ids = set()
    for update in answer["proposed_updates"]:
        if isinstance(update, dict) and isinstance(update.get("content"), dict):
            for node in update["content"].get("add_nodes", []):
                if isinstance(node, dict) and isinstance(node.get("id"), str):
                    proposed_ids.add(node["id"])
    evidence_ids = {obs["id"] for obs in public_input["visible_observations"]}
    for assessment in answer["aspect_assessments"]:
        if not isinstance(assessment, dict):
            errors.append("assessment_not_object")
            continue
        needed = {"target_ids", "evidence_ids", "availability", "assessment", "scope", "limitations", "rationale"}
        if not needed.issubset(assessment):
            errors.append("assessment_missing_fields")
        targets = assessment.get("target_ids", [])
        cited = assessment.get("evidence_ids", [])
        if not isinstance(targets, list) or not targets or any(not isinstance(value, str) or value not in known for value in targets):
            errors.append("invalid_assessment_targets")
        if not isinstance(cited, list) or any(not isinstance(value, str) or value not in evidence_ids for value in cited):
            errors.append("unknown_evidence_id")
        elif len(cited) != len(set(cited)):
            errors.append("duplicate_evidence_id")
        if assessment.get("availability") not in public_input["response_contract"].get("decision_definitions", {}).get("availability", {}):
            errors.append("availability_not_publicly_defined")
    for item in answer["findings"]:
        if not isinstance(item, dict) or not {"kind", "target_ids", "description", "basis_ids"}.issubset(item):
            errors.append("invalid_finding")
    for item in answer["verification_needs"]:
        if not isinstance(item, dict) or not {"target_ids", "unresolved_question", "needed_observations", "comparison_or_control", "why_informative", "missing_design_information"}.issubset(item):
            errors.append("invalid_verification_need")
    for field in ("findings", "proposed_updates", "verification_needs"):
        for item in answer[field]:
            if not isinstance(item, dict):
                errors.append(f"item_not_object:{field}")
                continue
            for key in ("target_ids", "basis_ids"):
                allowed_ids = known | proposed_ids if field == "verification_needs" and key == "target_ids" else known
                if key in item and (not isinstance(item[key], list) or any(not isinstance(value, str) or value not in allowed_ids for value in item[key])):
                    errors.append(f"unknown_or_invalid_reference:{field}:{key}")
    if any(not isinstance(value, str) or value not in known for value in answer["preserved_ids"]):
        errors.append("unknown_preserved_id")
    if not answer["proposed_updates"] and not answer["no_change_reason"]:
        errors.append("missing_no_change_reason")
    try:
        apply_proposed_updates(public_input, answer["proposed_updates"])
    except (ValueError, TypeError, KeyError) as exc:
        errors.append("unsafe_or_invalid_patch:" + str(exc))
    return sorted(set(errors))


def apply_proposed_updates(public_input: dict, proposed_updates: list[dict]) -> dict:
    """Apply the declared map-only patch vocabulary to an isolated copy.

    There is deliberately no arbitrary JSON Pointer mutation path. Even a
    proposed consolidation cannot delete scientific measurements or records.
    Source-span provenance of existing nodes and edges is immutable as well.
    """
    result = deepcopy(public_input)
    original_observations = stable_hash(public_input["visible_observations"])
    fields = {"add_nodes", "add_edges", "replace_nodes", "replace_edges", "remove_edge_ids"}
    for update in proposed_updates:
        if not isinstance(update, dict) or update.get("operation") not in OPERATIONS:
            raise ValueError("unknown_operation")
        if update["operation"] not in public_input.get("response_contract", {}).get("allowed_operations", []):
            raise ValueError("operation_not_allowed_by_public_contract")
        if not {"operation", "target_ids", "content", "basis_ids", "rationale"}.issubset(update):
            raise ValueError("update_missing_fields")
        if not isinstance(update["target_ids"], list) or any(not isinstance(identifier, str) for identifier in update["target_ids"]):
            raise ValueError("invalid_update_targets")
        declared_targets = set(update["target_ids"])
        patch = update["content"]
        if not isinstance(patch, dict) or not patch or set(patch) - fields:
            raise ValueError("only_explicit_map_patch_fields_allowed")
        if any(not isinstance(value, list) for value in patch.values()):
            raise ValueError("patch_fields_must_be_lists")
        if not any(patch.values()):
            raise ValueError("empty_patch")
        if update["operation"] in {"ADD_ASPECT", "DECOMPOSE_ASPECT", "PROPOSE_ALTERNATIVE"} and not patch.get("add_nodes"):
            raise ValueError("operation_requires_concrete_new_aspect")
        if update["operation"] == "REFINE_ASPECT" and not patch.get("replace_nodes"):
            raise ValueError("refinement_requires_replacement_wording")
        nodes = result["current_map"]["nodes"]
        edges = result["current_map"]["edges"]
        all_ids = object_ids(result)
        for collection, field in ((nodes, "add_nodes"), (edges, "add_edges")):
            for item in patch.get(field, []):
                if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"] or item["id"] in all_ids:
                    raise ValueError("new_id_collision_or_missing")
                if field == "add_nodes" and (item.get("type") not in ASPECT_TYPES | {"alternative", "mechanism", "proposed_mechanism"} or not item.get("statement")):
                    raise ValueError("new_node_must_be_concrete_proposed_aspect")
                if field == "add_edges" and not {"type", "source_id", "target_id"}.issubset(item):
                    raise ValueError("incomplete_new_relationship")
                collection.append(deepcopy(item))
                all_ids.add(item["id"])
        for collection, field in ((nodes, "replace_nodes"), (edges, "replace_edges")):
            positions = {item["id"]: index for index, item in enumerate(collection)}
            for item in patch.get(field, []):
                if not isinstance(item, dict) or item.get("id") not in positions:
                    raise ValueError("replacement_target_not_existing_map_object")
                existing = collection[positions[item["id"]]]
                if field == "replace_nodes" and item["id"] not in declared_targets:
                    raise ValueError("replacement_node_outside_declared_targets")
                if field == "replace_edges" and not declared_targets.intersection({item["id"], existing["source_id"], existing["target_id"]}):
                    raise ValueError("replacement_relationship_outside_declared_targets")
                if item.get("source_spans", []) != existing.get("source_spans", []):
                    raise ValueError("source_provenance_is_immutable")
                for key in ("source_ref", "provenance", "evidence_basis", "inference_reason"):
                    if key in item and item[key] != existing.get(key):
                        raise ValueError("source_provenance_is_immutable")
                if field == "replace_nodes" and (item.get("type") != existing.get("type") or item.get("type") not in ASPECT_TYPES or not item.get("statement")):
                    raise ValueError("only_existing_aspect_wording_can_be_refined")
                if field == "replace_edges" and not {"type", "source_id", "target_id"}.issubset(item):
                    raise ValueError("incomplete_replacement_relationship")
                # Do not silently discard scope, explicit requirements, or
                # provenance fields omitted by an otherwise well-formed patch.
                replacement = deepcopy(existing)
                replacement.update(deepcopy(item))
                collection[positions[item["id"]]] = replacement
        remove_ids = set(patch.get("remove_edge_ids", []))
        if not remove_ids.issubset({edge["id"] for edge in edges}):
            raise ValueError("removed_relationship_not_found")
        for edge in edges:
            if edge["id"] in remove_ids and not declared_targets.intersection({edge["id"], edge["source_id"], edge["target_id"]}):
                raise ValueError("removed_relationship_outside_declared_targets")
        result["current_map"]["edges"] = [edge for edge in edges if edge["id"] not in remove_ids]
        all_ids = object_ids(result)
        for edge in result["current_map"]["edges"]:
            if edge.get("source_id") not in all_ids or edge.get("target_id") not in all_ids:
                raise ValueError("dangling_relationship")
    if stable_hash(result["visible_observations"]) != original_observations:
        raise ValueError("observations_are_immutable")
    return result


def grade_answer(task: dict, answer: dict) -> dict:
    """Check a finite contract; unresolved alternatives route to adjudication.

    ``passed`` is a scoped checker outcome, never GRPO admission or a claim of
    scientific adequacy. The exporter enforces those separate review gates.
    """
    scoring = task["scoring"]
    result = {"status": "unqualified", "score": None, "errors": [], "scored_fields": scoring.get("scored_fields", []), "unscored_fields": scoring.get("unscored_fields", []), "scientific_semantics_certified": False}
    if scoring.get("grader_id") != GRADER_ID or scoring.get("grader_revision") != GRADER_REVISION or not scoring.get("sufficient_evidence_conditions"):
        result["errors"] = ["no_supported_finite_contract"]
        return result
    if scoring.get("scored_fields") != SCORED_FIELDS:
        result["errors"] = ["undeclared_or_unsupported_scored_fields"]
        return result
    errors = validate_answer(answer, task["public_input"])
    if errors:
        result.update(status="failed", score=0.0, errors=errors)
        return result
    decisions = {}
    for assessment in answer["aspect_assessments"]:
        for target in assessment["target_ids"]:
            if target in decisions:
                errors.append(f"duplicate_target_decision:{target}")
            decisions[target] = assessment
    unresolved = []
    conditions = scoring["sufficient_evidence_conditions"]
    target_ids = {condition["target_id"] for condition in conditions}
    if set(decisions) - target_ids:
        errors.append("assessment_outside_declared_finite_targets")
    for condition in conditions:
        target = condition["target_id"]
        decision = decisions.get(target)
        if decision is None:
            errors.append(f"missing_target_decision:{target}")
            continue
        if decision["availability"] != condition["availability"]:
            errors.append(f"incorrect_availability:{target}")
        selected = set(decision["evidence_ids"])
        allowed = set(condition["admissible_evidence_ids"])
        if selected - allowed:
            errors.append(f"irrelevant_evidence_padding:{target}")
            continue
        if not set(condition.get("required_evidence_ids", [])).issubset(selected):
            errors.append(f"required_conflict_or_evidence_omitted:{target}")
            continue
        alternatives = [set(item) for item in condition["sufficient_evidence_sets"]]
        permits_extra = condition.get("allow_additional_relevant_evidence", False)
        accepted = any((candidate.issubset(selected) if permits_extra else candidate == selected) for candidate in alternatives)
        if not accepted:
            if selected:
                unresolved.append(f"unreviewed_evidence_alternative:{target}")
            else:
                errors.append(f"insufficient_evidence_selection:{target}")
    if errors:
        result.update(status="failed", score=0.0, errors=sorted(set(errors)))
    elif unresolved:
        result.update(status="adjudication_required", score=None, errors=unresolved)
    else:
        result.update(status="passed", score=1.0)
    return result


def audit_grader(task: dict) -> dict:
    """Populate challenge payloads and run finite engineering checks in place.

    Answers here are synthetic checker fixtures, never SFT exemplars, scientific
    reviewer approvals, or intended-student samples. Comparator interpretation
    and contradictory prose are expressly outside this narrow finite checker.
    """
    conditions = task["scoring"].get("sufficient_evidence_conditions", [])
    if not conditions:
        return {"status": "not_applicable", "reason": "No finite contract."}
    base = {"aspect_assessments": [], "findings": [], "proposed_updates": [], "verification_needs": [], "preserved_ids": [], "information_requests": [], "no_change_reason": "Checker fixture preserves the map; no claim of scientific verification is made."}
    for condition in conditions:
        base["aspect_assessments"].append({
            "target_ids": [condition["target_id"]], "evidence_ids": condition["sufficient_evidence_sets"][0],
            "availability": condition["availability"], "assessment": "Availability inventory only; scientific adequacy is unscored.",
            "scope": condition.get("certified_claim", "Declared finite evidence contract."),
            "limitations": ["Does not establish scientific support."], "rationale": "Use the explicitly declared evidence inventory.",
        })
    valid = [{"name": "minimal_declared_evidence", "answer": deepcopy(base), "rationale": "Audited visibility can be established by an allowed minimal evidence set."}]
    invalid = []
    flipped = deepcopy(base)
    flipped["aspect_assessments"][0]["availability"] = "no_relevant_visible" if conditions[0]["availability"] == "relevant_visible" else "relevant_visible"
    invalid.append({"name": "flipped_availability", "answer": flipped, "rationale": "An availability claim inconsistent with the declared visible records must fail."})
    missing = deepcopy(base)
    missing["aspect_assessments"].pop()
    invalid.append({"name": "omitted_required_decision", "answer": missing, "rationale": "The public contract requires one decision per existing aspect."})
    observations = {obs["id"] for obs in task["public_input"]["visible_observations"]}
    for index, condition in enumerate(conditions):
        if len(condition["sufficient_evidence_sets"]) > 1:
            alternative = deepcopy(base)
            alternative["aspect_assessments"][index]["evidence_ids"] = condition["sufficient_evidence_sets"][1]
            valid.append({"name": "alternative_sufficient_evidence", "answer": alternative, "rationale": "A different explicitly permitted relevant subset must not be rejected for differing from the reference list."})
            break
    for index, condition in enumerate(conditions):
        if observations - set(condition["admissible_evidence_ids"]):
            padded = deepcopy(base)
            padded["aspect_assessments"][index]["evidence_ids"] = sorted(observations)
            invalid.append({"name": "select_all_irrelevant_padding", "answer": padded, "rationale": "Selecting all observations cannot bypass target-specific relevance constraints."})
            break
    task["scoring"]["valid_alternative_tests"] = valid
    task["scoring"]["invalid_answer_tests"] = invalid
    results = [{"name": item["name"], "expected_status": expected, "observed_status": grade_answer(task, item["answer"])["status"]} for collection, expected in ((valid, "passed"), (invalid, "failed")) for item in collection]
    # Apply a bijection to all object IDs in both public input and private
    # scoring, then check that the same scientific decision keeps its score.
    remapping = {identifier: f"v{index:03d}" for index, identifier in enumerate(sorted(object_ids(task["public_input"])), 1)}
    def permute(value: object) -> object:
        if isinstance(value, str):
            return remapping.get(value, value)
        if isinstance(value, dict):
            return {key: permute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [permute(item) for item in value]
        return value
    invariant = grade_answer(permute(task), permute(base))["status"] == grade_answer(task, base)["status"] == "passed"
    return {
        "status": "passed" if invariant and all(item["expected_status"] == item["observed_status"] for item in results) else "failed",
        "challenge_results": results, "id_permutation_invariant": invariant,
        "comparator_swap": "Not certified: this grader measures explicitly declared target-linked availability. Comparator or scope changes require source/task semantic re-review and invalidate dependent hashes.",
        "scientific_semantics_certified": False, "intended_student_sampled": False,
        "boundary": "Synthetic engineering challenges do not qualify this item for GRPO or provide a scientific reference answer.",
    }
