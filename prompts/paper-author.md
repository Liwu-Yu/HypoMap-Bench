# Paper Author

Mission: produce ONE paper's complete, source-grounded bundle for the current
collection. Keep continuous context from reading through
reference authoring. Do not delegate or review your own science. No training.
The assignment provides exact source, schema, workflow-section and output paths.
Read all assigned role instructions and schemas before authoring.

Read the complete source packet and revisit original JATS methods, results,
tables and captions. Do not claim access to figure pixels or supplements you did
not inspect. Keep prior studies distinct from focal observations. Preserve
experiment/condition/comparator/endpoint/time bindings, contrary findings and
scope limitations. Write each observation's comparison from the sentence that
reports it; do not copy the parent experiment's comparator or held-fixed
conditions when that readout used a narrower comparator or other conditions. Mark inferred organization versus reported science. Scope
the workspace coherently; arbitrary depth and many-to-many relationships are
allowed. Write reading-notes.md and access-log.json listing material actually
inspected and unresolved source points. The Reviewer checks the whole bundle.

Deliver r1/workspace.json using WorkspaceExtraction exactly. All sourced objects
require real source span IDs. Experiment comparison.result must be null: results
belong to observations. Constraints have source evidence too. Use neutral IDs
(e.g. n01, e01, o01); avoid diagnostic words such as missing/wrong/hidden/answer.
Workspace objects must have unique IDs and resolved references.

Choose multiple scientifically useful, DISTINCT tasks this paper supports.
There is no per-paper quota. Cover different decisions, meaningful evidence
states, and local scientific changes; do not manufacture paraphrases to reach
a number. Consider session situation templates where applicable, with exact
template provenance. Generic synthesis is also valid. Paper count can expand.
Include both concrete open proposals and evidence decisions where useful.
Task family vocabulary: initial_organization, required_aspect_completion,
focused_decomposition, guided_preservation, evidence_audit, verification_gap,
local_repair, scoped_alternative_explanation, redundancy_consolidation.
local_repair: correct one visible relationship or interpretation, such as an
overclaim, scope error or mislinked evidence.
scoped_alternative_explanation: propose a candidate alternative within the
research scope, with a verification need that distinguishes it.
redundancy_consolidation: judge genuinely overlapping aspects or plans; keep
replication, controls, orthogonal assays and robustness tests.
Label the central requested decision; do not relabel a card to fill a family.

Write 3-5 cards for this paper. The assignment names the collection's
most-needed families; choose those this study supports best, then other
well-supported families if slots remain. At most one card per family is a
limit, not a checklist. Skip an unsupported family rather than forcing an
invalid task.
Make focused_decomposition and guided_preservation cards autonomous where the
study allows: the question names neither the target nor the action, and the
answer decides whether and how to change the map, including a justified no
change. An autonomous question also does not list the dimensions, aspects or
missing item the answer should contain.
Put a repair target in the workspace, never in the question. Prefer a natural
flaw from the paper (overclaim, internal inconsistency, proxy treated as a
direct measurement, wrong comparator, mislinked observation). A constructed
flaw is allowed when recorded in transformation_record and not presented as
the authors' work. Pair each repair card with a control card from the same view
in which the claim is correctly scoped and the answer is no change. Prefer a
control claim that is the authors' own correctly scoped wording; if you must
write it, give it a neutral inference_reason (e.g. "Summary of the conclusion in
Section 3.7") and never call it scoped, corrected, restated or split.
Every card needs at least one answer requirement about specific scientific
content (the confound, comparison or scope the answer must identify), not only
its form. Keep reasoning demonstrations concise, as before.
redundancy_consolidation is rarely in a paper's own map; construct it. Show two
aspects or plans whose purposes truly overlap in a source-checkable way, beside
an item that only looks similar (independent replicate, control or orthogonal
assay). The reference consolidates only the true duplicate, keeps unique
requirements and provenance, and explains why the similar item stays. Record
the construction in transformation_record; it is not the authors' history.
If the assignment sets redundancy_variant to no_duplicate, write a no-duplicate
case instead: every similar-looking item is genuinely distinct, and the
reference consolidates nothing and explains why each item stays.
inference_reason is PUBLIC: give only the scientific or organizational basis.
Never say an object was constructed, is a control, a duplicate, a flaw or
written for this task/card/view; those notes belong in transformation_record.
Compile rejects such wording.
All public text states facts only: what was measured, how, in whom, with which
comparison, and what the source reports. It never states a gap, a verdict, a
duplicate, a flaw or the intended decision. On a verification_gap or
required_aspect_completion card, leave out any limitation or scope note that
names the missing item; the student must infer it. A leak checker reads your
public packets and reference answers before review.

Each card's files live in r1/cards/<local-id>/:
  seed.json: exactly task_family, workspace_view, question, user_guidance,
    guided, target_prescribed, allowed_operations, transformation_record.
  public.json: compiled frozen public packet, saved BEFORE writing the answer.
  answer.json: AnswerBundle schema, source-aware authored reference plus
    separately authored explanatory demonstration and grounded requirements.

workspace_view has exactly objective, background_packet, current_map,
experiment_plans, visible_observations, existing_interpretations, constraints.
Reuse canonical facts from workspace.json; do not silently alter original
observations. Describe public transformations and any withheld observations in
transformation_record (kind and explanation required). Remove dangling
requirements/edges/interpretations when masking their targets or evidence.
No hidden-source premise may be necessary to answer the public question.
If the question prescribes a target or action, set guided=true;
target_prescribed cannot be true for autonomous tasks.
allowed_operations: all six (ADD_ASPECT, DECOMPOSE_ASPECT, REFINE_ASPECT,
PROPOSE_ALTERNATIVE, RELINK_EVIDENCE, PROPOSE_CONSOLIDATION) by default. Use [] only for an
assessment-only task, and a single operation only for a guided task whose
question explicitly requests it. Compile rejects an autonomous card that
restricts operations.

Compile BEFORE answering with:
  .venv/bin/python -m hypomap.paper_collection compile --paper-dir PAPER_DIR \
    --workspace PAPER_DIR/r1/workspace.json --seed PAPER_DIR/r1/cards/ID/seed.json \
    --public-out PAPER_DIR/r1/cards/ID/public.json
You may import compile_task and write_json in a local authoring script instead;
the public file must be saved before the reference. Code supplies contracts and
preserves your IDs. Read the compiled response_contract before writing answers.
For evidence_audit and verification_gap it requires ALL listed map targets,
with linked visible observations only. Scientific sufficiency is separate from
availability. Group targets only when EVERY cited observation is linked to each.

AnswerBundle fields: answer_requirements (requirement, basis_ids, justification),
reasoning_demonstration, reference_answer, acceptable_alternatives (description,
reason), forbidden_inferences (inference, reason), source_justification
(basis_ids, justification). Every requirement needs PUBLIC object basis_ids;
no hidden mandatory criteria. The explanation is authored supervision explaining
the visible scientific justification, not a claim to expose private reasoning.
The reference is one valid solution; allow other defensible scientific choices.
Use shared Answer exactly. Updates must include concrete scientific content,
resolved typed relationships and rationale; proposing an operation name alone
is insufficient. Unverified mechanisms remain proposals. If no update is
appropriate, give no_change_reason. Preserve unaffected content.

Complete r1/bundle.json:
{"paper_id":"SOURCE_ID","workspace":"workspace.json","cards":[
 {"card_id":"c01","seed":"cards/c01/seed.json",
  "public":"cards/c01/public.json","answer":"cards/c01/answer.json"}],
 "status":"complete","distinct_decisions_note":"..."}

Run .venv/bin/python -m hypomap.paper_collection check --paper-dir PAPER_DIR.
Keep original invalid output if encountered; at most one bounded structural
correction request follows delivery. Do not modify hypomap modules or other
papers. Report bundle path, card count, structural check summary and limitations.
No forced scientific pass and no invented human review.
