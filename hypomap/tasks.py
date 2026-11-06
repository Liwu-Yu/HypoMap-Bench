"""Construct frozen, source-linked pilot tasks; scientific review is a later stage."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re


WORKSPACE_FIELDS = (
    "objective", "background_packet", "current_map", "experiment_plans",
    "visible_observations", "existing_interpretations", "constraints",
)
PUBLIC_FIELDS = (*WORKSPACE_FIELDS, "user_guidance", "question", "response_contract")
OUTPUT_FIELDS = (
    "aspect_assessments", "findings", "proposed_updates", "verification_needs",
    "preserved_ids", "information_requests", "no_change_reason",
)
OPERATIONS = (
    "ADD_ASPECT", "DECOMPOSE_ASPECT", "REFINE_ASPECT", "PROPOSE_ALTERNATIVE",
    "RELINK_EVIDENCE", "PROPOSE_CONSOLIDATION",
)
PRIVATE_KEYS = {
    "private_answer", "answer_requirements", "reference_answer", "acceptable_alternatives",
    "forbidden_inferences", "transformation_record", "neutral_id_map", "permitted_uses",
    "task_family", "construction_origin", "intended_operation", "review", "identifiers",
    "source_map_hash", "hidden_observations", "correct_answer", "expected_answer",
    "token_accounting_private", "training_use_allocation",
    "scope_selection_record", "compression_fidelity_review", "answerability_review_after_rendering",
    "token_review", "rendered_prompt_tokens", "reference_completion_tokens", "exact_sft_sequence_tokens",
    "packet_compaction", "span_aliases", "pre_compaction_public_input",
    "private_supervision", "rendering_private", "review_private", "reasoning_demonstration",
    "reasoning_provenance_and_review", "reward_spec",
}
ASPECT_TYPES = {"aspect", "validation_aspect", "sub_hypothesis", "subhypothesis", "hypothesis", "claim"}
PUBLIC_CONTRACT_REVISION = "scientific-workspace-response-v2"


def finite_target_inventory(packet: dict) -> list[dict]:
    """A public map checklist, with neither availability labels nor hidden data."""
    return [{"target_id": node["id"], "node_type": node["type"]}
            for node in packet["current_map"]["nodes"] if node.get("type") in ASPECT_TYPES]


def operation_requirements() -> dict:
    """Publish the existing patch checker's operation-specific conditions."""
    return {
        operation: {"required_nonempty_patch_fields": (
            ["add_nodes"] if operation in {"ADD_ASPECT", "DECOMPOSE_ASPECT", "PROPOSE_ALTERNATIVE"}
            else ["replace_nodes"] if operation == "REFINE_ASPECT" else [])}
        for operation in OPERATIONS
    }


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def workspace_content_hash(workspace: dict) -> str:
    return stable_hash({key: workspace.get(key) for key in WORKSPACE_FIELDS})


def public_input_hash(task: dict) -> str:
    return stable_hash(task["public_input"])


def answer_contract_hash(task: dict) -> str:
    return stable_hash(task["private_answer"])


def scoring_contract_hash(task: dict) -> str:
    return stable_hash(task["scoring"])


def dependency_hashes(task: dict) -> dict:
    return {
        "source_map_hash": task["lineage"]["source_map_hash"],
        "public_input_hash": public_input_hash(task),
        "answer_contract_hash": answer_contract_hash(task),
        "scoring_contract_hash": scoring_contract_hash(task),
    }


def pending_token_accounting() -> dict:
    """Required private section, deliberately unqualified until exact rendering."""
    return {
        "model_id": "Qwen/Qwen3-8B", "model_revision": None, "tokenizer_revision": None,
        "chat_template_hash": None, "serializer_hash": None, "enable_thinking": True,
        "supervision_mode": "thinking", "native_sequence_capacity_tokens": 32768,
        "max_prompt_tokens": None, "max_sequence_tokens": 32768,
        "generation_budget_tokens": 8192, "reserved_completion_tokens": 8192,
        "generation_budget_profile": None, "generation_budget_policy_id": None,
        "generation_budget_policy_sha256": None, "generation_budget_public_inputs": {},
        "rendered_prompt_tokens": None, "reference_completion_tokens": None,
        "reasoning_tokens": None, "final_answer_tokens": None, "generated_delimiter_tokens": None,
        "reasoning_demo_sha256": None, "reasoning_final_boundary_verified": False,
        "exact_sft_sequence_tokens": None, "input_truncation_detected": False,
        "reference_is_complete": None, "scope_selection_record": {},
        "compression_fidelity_review": {}, "answerability_review_after_rendering": {},
    }


def _objects(packet: dict) -> list[dict]:
    return [packet.get("objective", {})] + list(packet.get("objective", {}).get("validation_requirements", [])) + [
        obj for field in ("background_packet", "experiment_plans", "visible_observations", "existing_interpretations", "constraints")
        for obj in packet.get(field, [])
    ] + packet.get("current_map", {}).get("nodes", []) + packet.get("current_map", {}).get("edges", [])


def object_ids(packet: dict) -> set[str]:
    return {obj["id"] for obj in _objects(packet) if isinstance(obj, dict) and isinstance(obj.get("id"), str)}


def _strip_private(value: object) -> object:
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items() if key not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def _neutralize(packet: dict, id_map: dict | None = None) -> tuple[dict, dict]:
    # Type-free IDs reveal neither a transformation nor the target answer. The
    # mapping remains in the private lineage, never in a model-facing export.
    id_map = id_map or {obj["id"]: f"i{index:03d}" for index, obj in enumerate(_objects(packet), 1) if obj.get("id")}
    reference_fields = {"id", "source_id", "target_id", "experiment_id", "aspect_ids", "target_aspect_ids", "basis_ids", "derived_from_ids"}
    def walk(value: object, field: str | None = None) -> object:
        if isinstance(value, str):
            # An identifier such as "stability" can also be a scientific word.
            # Only typed ID/reference fields are aliases; prose and source
            # locator strings retain their original scientific meaning.
            return id_map.get(value, value) if field in reference_fields else value
        if isinstance(value, list):
            return [walk(item, field) for item in value]
        if isinstance(value, dict):
            return {key: walk(item, key) for key, item in value.items()}
        return value
    return walk(_strip_private(packet)), id_map


def _safe_background(items: list[dict], hidden: list[dict]) -> list[dict]:
    hidden_spans = {span for obj in hidden for span in obj.get("source_spans", [])}
    safe = []
    for item in items:
        if hidden_spans.intersection(item.get("source_spans", [])):
            continue
        if item.get("role") in {"definition", "assumption"}:
            safe.append(item)
        elif item.get("role") == "prior_finding":
            applicability = item.get("applicability", {})
            if applicability.get("origin") in {"external_study", "prior_study", "other_study"}:
                safe.append(item)
    return safe


def _organization_view(full: dict) -> dict:
    """Construct an organization state without copying known result summaries."""
    packet = deepcopy(full)
    hidden = packet["visible_observations"]
    hidden_spans = {span for obj in hidden for span in obj.get("source_spans", [])}
    packet["experiment_plans"] = []
    packet["visible_observations"] = []
    packet["existing_interpretations"] = []
    packet["background_packet"] = _safe_background(packet["background_packet"], hidden)
    packet["constraints"] = [obj for obj in packet["constraints"] if not hidden_spans.intersection(obj.get("source_spans", []))]
    # Methods/observations are omitted rather than relabeled as unperformed.
    remaining = object_ids(packet)
    packet["current_map"]["edges"] = [edge for edge in packet["current_map"]["edges"]
        if edge["source_id"] in remaining and edge["target_id"] in remaining]
    return packet


def _response_contract() -> dict:
    return {
        "contract_revision": PUBLIC_CONTRACT_REVISION,
        "answer_mode": "open_scientific",
        "allowed_operations": list(OPERATIONS),
        "operation_requirements": operation_requirements(),
        "operation_interpretation": "Every operation requires a nonempty map patch. REFINE_ASPECT requires replacement wording in replace_nodes for an existing declared target; adding a refinement child alone does not satisfy it. A new child can instead be proposed with ADD_ASPECT or DECOMPOSE_ASPECT. RELINK_EVIDENCE and PROPOSE_CONSOLIDATION have no additional required nonempty field; all shared patch rules still apply. No operation deletes map nodes or changes observations.",
        "required_output_fields": list(OUTPUT_FIELDS),
        "decision_definitions": {
            "availability": {
                "relevant_visible": "At least one supplied observation explicitly addresses the target aspect at its recorded scope; this does not establish sufficiency or verification.",
                "no_relevant_visible": "No supplied observation addresses the target aspect; this says nothing about unprovided measurements or historical execution.",
            },
            "assessment": "A scoped scientific assessment in words, including uncertainty; no aggregate all-aspects rule is implied.",
        },
        "empty_collections": "An empty collection means this packet provides no records of that kind, not that the source study never performed the work.",
        "patch_shape": {
            "add_nodes": "list of new typed aspect nodes with id, type, statement",
            "add_edges": "list of typed edges with id, type, source_id, target_id",
            "replace_nodes": "list of existing aspect nodes with replacement statement and unchanged source_spans",
            "replace_edges": "list of existing edges with replacement connection and unchanged source_spans",
            "remove_edge_ids": "list of existing relationship IDs; source records and observations cannot be changed",
        },
        "patch_rules": {
            "refinable_node_types": sorted(ASPECT_TYPES),
            "shape": "Use only the five patch_shape fields, each a list. At least one list must be nonempty. operation must be in allowed_operations. Supply target_ids, basis_ids and a rationale for every update.",
            "new_objects": "New node and edge IDs must be nonempty and unique across all packet objects and preceding updates. Node types must satisfy the final-answer JSON Schema, with nonempty statements; edges require type, source_id and target_id. Relationship endpoints must resolve after each update.",
            "replacement_nodes": "Only existing nodes of refinable_node_types can have wording replaced, subject to the final-answer JSON Schema. Keep their ID, node type and source_spans unchanged, give a nonempty replacement statement, and include every replaced node ID in that update's target_ids.",
            "replacement_edges": "Replace only existing edges and keep their ID and source_spans. Include the existing edge ID or at least one of its existing endpoint IDs in that update's target_ids. Supply type, source_id and target_id for the replacement.",
            "removed_edges": "Remove only existing edges. Include each removed edge ID or at least one of its endpoint IDs in that update's target_ids. Edges added earlier in the same patch must satisfy the same rule if removed.",
            "provenance": "Existing source_ref, provenance, evidence_basis and inference_reason cannot be changed when supplied in a replacement. Omitted existing fields remain unchanged. Source records and observations are immutable.",
            "references": "Assessment evidence_ids must name visible observations without duplicates. Assessment targets, findings, update target_ids/basis_ids and preserved_ids must resolve to the original public packet. Verification targets may also name validated newly proposed nodes. When proposing no updates, supply no_change_reason.",
        },
        "preservation": "Unmentioned content is unchanged. Proposals apply to a copy for checking. Measurements and source provenance are immutable. New scientific content is proposed, not observed.",
    }


def _make_task(workspace: dict, packet: dict, family: str, transformation: dict,
               *, guided: bool = False, parent_ids: list[str] | None = None,
               id_map: dict | None = None, guidance: str | None = None,
               question: str | None = None, target_prescribed: bool = False,
               allowed_operations: list[str] | None = None,
               public_contract_options: dict[str, bool] | None = None) -> dict:
    options = {} if public_contract_options is None else public_contract_options
    if (not isinstance(options, dict)
            or set(options) - {"show_scored_fields", "show_inventory_construction_note"}
            or any(type(value) is not bool for value in options.values())):
        raise ValueError("public_contract_options supports only the two named boolean options")
    packet, neutral_ids = _neutralize(packet, id_map)
    packet.update(user_guidance=None, response_contract=_response_contract())
    packet["question"] = (
        "Organize the supplied objective and its explicit validation requirements into concrete, separately testable aspects and their relationships. Preserve the stated scientific scope, identify missing information, and describe verification needs without inventing observations."
        if guided else
        "Review this scientific workspace. Assess what the available evidence does and does not establish, identify unresolved questions, and recommend a justified next validation step. Choose any warranted local proposals, preservation, or requests for missing information; explain their visible basis."
    )
    if guidance is not None:
        packet["user_guidance"] = guidance
    if question is not None:
        packet["question"] = question
    if allowed_operations is not None:
        packet["response_contract"]["allowed_operations"] = list(allowed_operations)
    if family in {"evidence_audit", "verification_gap"}:
        packet["response_contract"]["answer_mode"] = "mixed"
        construction_note = "The required_targets checklist is derived from all existing public map nodes of assess_all_node_types, including claims. "
        packet["response_contract"]["finite_assessment_scope"] = {
            "assess_all_node_types": sorted(ASPECT_TYPES),
            "required_targets": finite_target_inventory(packet),
            "instruction": construction_note + "Cover every listed target exactly once and no target outside that checklist; do not narrow it to nodes literally typed aspect. Use observations' explicit target_aspect_ids: cite at least one linked observation when available, only linked observations, and none when unavailable. Group targets in one assessment only if its availability and every cited observation satisfy the rule for each grouped target; a union of separately relevant observations is not sufficient. Interpret scientific sufficiency separately in assessment and limitations. Which local update to propose, and its target, remain your decisions.",
            "scored_fields": ["aspect_assessments.target_ids", "aspect_assessments.evidence_ids", "aspect_assessments.availability"],
            "boundary": "This finite inventory decision does not certify comparator adequacy, causal conclusions, or scientific correctness of prose.",
        }
        scope = packet["response_contract"]["finite_assessment_scope"]
        if not options.get("show_scored_fields", True):
            del scope["scored_fields"]
        if not options.get("show_inventory_construction_note", True):
            scope["instruction"] = scope["instruction"].removeprefix(construction_note)
    identifiers = workspace["identifiers"]
    source_ids = identifiers.get("source_ids") or [identifiers["source_id"]]
    seed = ["source-grounded-task-v3", identifiers["background_id"], identifiers.get("revision_id"), family, transformation, stable_hash(packet)]
    task_id = "task-" + stable_hash(seed)[:20]
    transformation = deepcopy(transformation)
    transformation.update(neutral_id_map=neutral_ids, frozen_public_input_hash=stable_hash(packet))
    task = {
        "identifiers": {
            "task_id": task_id, "revision_id": "r1-" + stable_hash(seed)[:12],
            "background_id": identifiers["background_id"], "source_family_id": identifiers["source_family_id"],
            "source_ids": source_ids, "split": identifiers["split"],
        },
        "lineage": {
            "construction_origin": "generic_synthesis", "session_event_ids": [], "situation_template_ids": [],
            "parent_task_ids": parent_ids or [], "transformation_record": transformation,
            "source_map_hash": workspace_content_hash(workspace),
        },
        "public_input": packet,
        "token_accounting_private": pending_token_accounting(),
        "capability": {"task_family": family, "decision_mode": "guided_execution" if guided else "autonomous_decision", "answer_mode": packet["response_contract"]["answer_mode"], "action_prescribed": guided, "target_prescribed": target_prescribed},
        "private_answer": {"answer_requirements": [], "reference_answer": None, "reasoning_demonstration": None, "acceptable_alternatives": [], "forbidden_inferences": [], "source_justification": []},
        "scoring": {"grader_id": None, "grader_revision": None, "scored_fields": [], "unscored_fields": list(OUTPUT_FIELDS), "sufficient_evidence_conditions": [], "invalid_answer_tests": [], "valid_alternative_tests": []},
        "review": {
            "source_reviews": deepcopy(workspace.get("review", {}).get("source_reviews", [])),
            "task_reviews": [], "frozen_blind_answer": None, "answer_adjudication": [], "grader_reviews": [], "review_provenance": [],
            "permitted_uses": ["analysis"], "use_justifications": {"analysis": "Constructed source-linked candidate; scientific task and answer review are pending."},
            "training_use_allocation": {"stages": [{"train": "sft", "development": "development", "test": "final_test"}[identifiers["split"]]], "allow_task_reuse": False, "reuse_justification": None},
        },
    }
    if family in {"evidence_audit", "verification_gap"}:
        from .grading import availability_scoring_contract
        task["scoring"] = availability_scoring_contract(task)
    return task


def construct_tasks(workspaces: list[dict], templates: list[dict]) -> list[dict]:
    """Construct source-supported states, never enough variants just to fill a quota.

    Whole-workspace source review remains an upstream prerequisite. Open tasks
    and bounded inventory candidates are separate capabilities; neither enters
    a training export solely because this constructor can make a task card.
    """
    tasks = []
    for workspace in workspaces:
        full = deepcopy({field: workspace[field] for field in WORKSPACE_FIELDS})
        _, id_map = _neutralize(full)
        aspects = [node for node in full["current_map"]["nodes"] if node.get("type") in ASPECT_TYPES]
        observations = full["visible_observations"]
        if len(full["objective"].get("validation_requirements", [])) >= 2:
            initial = _organization_view(full)
            initial["current_map"] = {"nodes": [], "edges": []}
            for requirement in initial["objective"].get("validation_requirements", []):
                requirement["aspect_ids"] = []
            removed_ids = sorted(object_ids(full) - object_ids(initial))
            modified_ids = {item["id"] for item in initial["objective"].get("validation_requirements", [])}
            tasks.append(_make_task(workspace, initial, "initial_organization", {
                "kind": "initial_organization_view", "removed_object_ids": removed_ids,
                "modified_object_ids": sorted(modified_ids),
                "unchanged_object_ids": sorted(object_ids(initial) - modified_ids),
                "interpretation": "A constructed initial view, not the authors' historical workflow.",
            }, guided=True, id_map=id_map,
                question="Organize the objective and its supplied validation requirements into concrete, separately testable questions and relationships. The requirements are given; their organization and scientific wording are open. Preserve the stated scope, explain alternatives and needed verification, and do not invent observations or attempt to recover an unseen paper hierarchy."))
            # Omit a leaf linked to an explicit requirement, retaining the
            # requirement as the public reason a question may need adding.
            # No hidden experiment or outcome is an answer requirement.
            for aspect in aspects:
                children = [edge for edge in full["current_map"]["edges"]
                            if edge["source_id"] == aspect["id"] and edge.get("type") in {"composition", "refinement"}]
                explicit = [requirement for requirement in full["objective"].get("validation_requirements", [])
                            if requirement.get("aspect_ids") == [aspect["id"]]]
                if children or not explicit or len(aspects) < 2:
                    continue
                completion = _organization_view(full)
                completion["current_map"]["nodes"] = [node for node in completion["current_map"]["nodes"] if node["id"] != aspect["id"]]
                completion["current_map"]["edges"] = [edge for edge in completion["current_map"]["edges"]
                    if aspect["id"] not in {edge["source_id"], edge["target_id"]}]
                for requirement in completion["objective"].get("validation_requirements", []):
                    requirement["aspect_ids"] = [identifier for identifier in requirement.get("aspect_ids", []) if identifier != aspect["id"]]
                modified_ids = {item["id"] for item in full["objective"].get("validation_requirements", []) if aspect["id"] in item.get("aspect_ids", [])}
                tasks.append(_make_task(workspace, completion, "required_aspect_completion", {
                    "kind": "omit_explicitly_required_leaf_aspect", "omitted_aspect_id": aspect["id"],
                    "public_requirement_ids": [item["id"] for item in explicit],
                    "removed_object_ids": sorted(object_ids(full) - object_ids(completion)),
                    "modified_object_ids": sorted(modified_ids),
                    "unchanged_object_ids": sorted(object_ids(completion) - modified_ids),
                    "interpretation": "Synthetic organization state. A supplied requirement warrants checking map coverage; multiple additions or refinements can be valid, and hidden outcomes are not required.",
                }, id_map=id_map,
                    question="Review whether the current questions adequately represent the objective's supplied validation requirements. Decide whether an addition, a refinement, preservation, or clarification is warranted. Provide concrete scientific wording and relations for any proposal, preserve useful distinctions, and explain the visible requirement that justifies it. Missing observation records alone are not a reason to add a duplicate aspect."))
                break
        # A session template supplies the request situation only. Scientific
        # statements and nested relationships continue to come from the paper.
        safe_template = next((template for template in templates
            if (template.get("pattern") == "guided_decomposition_request"
                    and template.get("contains_participant_text") is False
                    and template.get("human_preference_inferred") is False
                    and template.get("supporting_event_ids"))), None)
        if full["objective"].get("validation_requirements") or observations:
            target = None
            for aspect in aspects:
                children = {edge["target_id"] for edge in full["current_map"]["edges"]
                            if edge["source_id"] == aspect["id"]
                            and edge.get("type") in {"refinement", "composition", "refines", "composes", "has_aspect", "decomposes_into"}}
                dimensions = [requirement for requirement in full["objective"].get("validation_requirements", []) if aspect["id"] in requirement.get("aspect_ids", [])]
                if len(children) >= 2 or len(dimensions) >= 2:
                    target = aspect
                    break
            if target is not None:
                public_target = id_map[target["id"]]
                guided_task = _make_task(workspace, full, "focused_decomposition", {
                    "kind": "abstract_guided_refinement_request", "removed_object_ids": [],
                    "unchanged_object_ids": sorted(object_ids(full)), "target_aspect_id": target["id"],
                    "interpretation": "A synthetic request to examine separately assessable dimensions; existing organization may be preserved. It is not a claim that the source hierarchy is defective or uniquely optimal.",
                }, guided=True, id_map=id_map, target_prescribed=True,
                    guidance=f"Please review and refine aspect {public_target} into separately assessable questions, preserving its scope and any existing useful distinctions.",
                    question=f"Carry out the supplied refinement request for {public_target}. Include concrete wording and relationships for any proposed changes. If the existing organization already distinguishes the relevant questions, explain preservation or request the information needed to refine further; do not invent observations.")
                if safe_template:
                    guided_task["lineage"].update(construction_origin="session_informed_synthesis", session_event_ids=list(safe_template["supporting_event_ids"]), situation_template_ids=[safe_template["template_id"]])
                tasks.append(guided_task)
        if len(aspects) >= 2 and observations:
            preservation = _make_task(workspace, full, "guided_preservation", {
                "kind": "preserve_structure_assess_remaining_validation", "removed_object_ids": [],
                "unchanged_object_ids": sorted(object_ids(full)),
                "interpretation": "A synthetic instruction to retain the supplied map while evaluating evidence and verification needs; not autonomous stopping or proof that validation is complete.",
            }, guided=True, target_prescribed=True, id_map=id_map, allowed_operations=[],
                guidance="Keep the existing map structure and recorded observations unchanged for this assessment. Identify useful preserved distinctions and any unresolved verification or information needs.",
                question="Within the supplied preservation instruction, assess what the visible evidence establishes at its actual comparison scope. Explain which distinctions, controls, or measurement scopes should be retained and why. State concrete verification needs or necessary information requests without claiming that preservation proves complete validation. Return no structural patch; a scientific concern can be described and investigated without editing the current map.")
            # This is genuinely a prescribed preservation task, not an
            # autonomous task with a hidden zero-edit reward.
            tasks.append(preservation)
        target_patterns = {tuple(sorted(obs.get("target_aspect_ids", []))) for obs in observations}
        evidence_kinds = {obs.get("kind") for obs in observations}
        if len(observations) < 2 or len(aspects) < 2 or (len(target_patterns) < 2 and len(evidence_kinds) < 2):
            continue
        audit = _make_task(workspace, full, "evidence_audit", {
            "kind": "full_reported_workspace", "removed_object_ids": [],
            "unchanged_object_ids": sorted(object_ids(full)),
            "interpretation": "A reconstructed paper workspace, not an observed author decision.",
        }, id_map=id_map)
        tasks.append(audit)
        # Withhold a complete target's recorded evidence while retaining evidence
        # elsewhere. Do not mask a value and demand that the solver recover it.
        for aspect in aspects:
            removed = [obs for obs in observations if aspect["id"] in obs.get("target_aspect_ids", [])]
            if not removed or len(removed) == len(observations):
                continue
            partial = deepcopy(full)
            hidden_ids = {obs["id"] for obs in removed}
            # Remove downstream computed results that directly cite hidden data.
            changed = True
            while changed:
                changed = False
                for obs in observations:
                    dependencies = set(obs.get("basis_ids", []) + obs.get("derived_from_ids", []))
                    if obs["id"] not in hidden_ids and dependencies.intersection(hidden_ids):
                        hidden_ids.add(obs["id"])
                        changed = True
            removed = [obs for obs in observations if obs["id"] in hidden_ids]
            partial["visible_observations"] = [obs for obs in partial["visible_observations"] if obs["id"] not in hidden_ids]
            if not partial["visible_observations"]:
                continue
            partial["existing_interpretations"] = []
            partial["background_packet"] = _safe_background(partial["background_packet"], removed)
            hidden_spans = {span for obj in removed for span in obj.get("source_spans", [])}
            partial["constraints"] = [obj for obj in partial["constraints"] if not hidden_spans.intersection(obj.get("source_spans", []))]
            visible_ids = object_ids(partial)
            partial["current_map"]["edges"] = [edge for edge in partial["current_map"]["edges"] if edge["source_id"] in visible_ids and edge["target_id"] in visible_ids]
            # Source-map nodes are claims, aspects, mechanisms, alternatives or
            # limitations, never duplicate observation records. Preserve their
            # scientific roles and every reference from requirements/methods.
            # Whether a node's wording leaks a withheld finding requires source-
            # aware semantic review, not deletion based on its node type.
            partial_task = _make_task(workspace, partial, "verification_gap", {
                "kind": "withhold_recorded_observations", "removed_observation_ids": sorted(hidden_ids),
                "removed_object_ids": sorted(object_ids(full) - object_ids(partial)),
                "unchanged_object_ids": sorted(object_ids(partial)), "target_aspect_id": aspect["id"],
                "interpretation": "Synthetic availability change; hidden directions and values are not answer requirements.",
            }, parent_ids=[audit["identifiers"]["task_id"]], id_map=id_map)
            for template in templates:
                if (template.get("pattern") == "existing_aspect_missing_measurement"
                        and template.get("safe_for_external_use") is True
                        and template.get("supporting_event_ids")
                        and template.get("applicability_conditions") == {"existing_aspect": True, "no_visible_measurement_for_aspect": True, "other_visible_observations": True}):
                    partial_task["lineage"]["construction_origin"] = "session_informed_synthesis"
                    partial_task["lineage"]["session_event_ids"] = list(template["supporting_event_ids"])
                    partial_task["lineage"]["situation_template_ids"] = [template["template_id"]]
                    break
            tasks.append(partial_task)
            break  # One controlled pair per applicable background in the pilot.
    return tasks


def validate_task(task: dict, workspace: dict | None = None) -> list[str]:
    """Structural and dependency checks; passing is not scientific validation."""
    errors = []
    required = {"identifiers", "lineage", "public_input", "capability", "private_answer", "scoring", "review", "token_accounting_private"}
    if set(task) != required:
        errors.append("canonical_top_level_fields_mismatch")
    if not required.issubset(task):
        return errors
    packet = task["public_input"]
    if set(packet) != set(PUBLIC_FIELDS):
        errors.append("public_fields_mismatch")
    for field in ("background_packet", "experiment_plans", "visible_observations", "existing_interpretations", "constraints"):
        if not isinstance(packet.get(field), list):
            errors.append(f"collection_not_list:{field}")
    if errors:
        return errors
    if task["identifiers"].get("split") not in {"train", "development", "test"}:
        errors.append("invalid_split")
    ids = [obj.get("id") for obj in _objects(packet)]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        errors.append("missing_object_id")
    if len(ids) != len(set(ids)):
        errors.append("duplicate_object_id")
    contract = packet.get("response_contract", {})
    # Historical frozen packets retain their old schema. New packets cannot
    # publish a narrowed checklist or hide operation conditions from students.
    if contract.get("contract_revision") == PUBLIC_CONTRACT_REVISION:
        if contract.get("operation_requirements") != operation_requirements():
            errors.append("public_operation_requirements_mismatch")
        if contract.get("answer_mode") in {"mixed", "finite_decision"}:
            scope = contract.get("finite_assessment_scope", {})
            inventory = finite_target_inventory(packet)
            if (scope.get("assess_all_node_types") != sorted(ASPECT_TYPES)
                    or scope.get("required_targets") != inventory):
                errors.append("public_finite_target_inventory_mismatch")
            expected = [item["target_id"] for item in inventory]
            actual = [item.get("target_id") for item in task["scoring"].get("sufficient_evidence_conditions", [])]
            if actual != expected:
                errors.append("finite_scoring_target_inventory_mismatch")
    known = set(ids)
    for edge in packet.get("current_map", {}).get("edges", []):
        if edge.get("source_id") not in known or edge.get("target_id") not in known:
            errors.append(f"dangling_edge:{edge.get('id')}")
    for obj in _objects(packet):
        for field in ("aspect_ids", "target_aspect_ids", "basis_ids", "derived_from_ids"):
            for target in obj.get(field, []):
                if target not in known:
                    errors.append(f"dangling_reference:{obj.get('id')}:{field}:{target}")
        if obj.get("experiment_id") and obj["experiment_id"] not in known:
            errors.append(f"dangling_experiment:{obj.get('id')}")
    def inspect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in PRIVATE_KEYS:
                    errors.append(f"private_key_in_public:{key}")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    inspect(packet)
    if any(re.search(r"(?i)(missing|combined|wrong|hidden|repair|answer)", identifier) for identifier in ids if isinstance(identifier, str)):
        errors.append("diagnostic_public_id")
    frozen = task["lineage"].get("transformation_record", {}).get("frozen_public_input_hash")
    if frozen != public_input_hash(task):
        errors.append("public_packet_changed_since_freeze")
    capability = task["capability"]
    if (capability.get("answer_mode") not in {"open_scientific", "finite_decision", "mixed"}
            or capability.get("answer_mode") != packet.get("response_contract", {}).get("answer_mode")):
        errors.append("capability_answer_mode_mismatch")
    if capability.get("decision_mode") == "autonomous_decision" and (capability.get("action_prescribed") or capability.get("target_prescribed")):
        errors.append("autonomous_task_has_prescribed_choice")
    if workspace is not None:
        if task["lineage"].get("source_map_hash") != workspace_content_hash(workspace):
            errors.append("source_workspace_changed")
        for key in ("source_family_id", "background_id", "split"):
            if task["identifiers"].get(key) != workspace["identifiers"].get(key):
                errors.append(f"workspace_identity_mismatch:{key}")
        transform = task["lineage"]["transformation_record"]
        compaction = transform.get("packet_compaction")
        if compaction:
            aliases = compaction.get("span_aliases", {})
            if (not isinstance(aliases, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in aliases.items())
                    or len(aliases) != len(set(aliases.values()))):
                errors.append("invalid_or_noninjective_span_aliases")
            else:
                source_spans = set()
                def collect(value: object) -> None:
                    if isinstance(value, dict):
                        source_spans.update(value.get("source_spans", []))
                        for item in value.values():
                            collect(item)
                    elif isinstance(value, list):
                        for item in value:
                            collect(item)
                collect({field: workspace[field] for field in WORKSPACE_FIELDS})
                if not set(aliases).issubset(source_spans):
                    errors.append("span_alias_original_not_in_source_workspace")
                inverse = {value: key for key, value in aliases.items()}
                def expand(value: object) -> object:
                    if isinstance(value, str):
                        return inverse.get(value, value)
                    if isinstance(value, list):
                        return [expand(item) for item in value]
                    if isinstance(value, dict):
                        return {key: expand(item) for key, item in value.items()}
                    return value
                restored = expand(packet)
                original = task["review"].get("pre_compaction_public_input")
                if (compaction.get("aliased_public_input_hash") != public_input_hash(task)
                        or compaction.get("original_public_input_hash") != stable_hash(restored)
                        or not isinstance(original, dict) or original != restored):
                    errors.append("source_locator_compaction_not_lossless")
        mapping = transform.get("neutral_id_map", {})
        for hidden in transform.get("removed_observation_ids", []):
            if mapping.get(hidden, hidden) in known:
                errors.append(f"hidden_observation_visible:{hidden}")
    return sorted(set(errors))
