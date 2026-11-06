# Decision-variant Author addendum

Read this after `paper-author.md`. `docs/specs/decision-variants.md` §2–4
defines the design. You handle one copied paper, using gpt-6.1-sol / high.

Read whatever source the construction needs. Do not re-audit unchanged
reviewed material. Record the spans you inspected.

## D1 cards and the necessity standard

Only D1 cards (spec §2) get pairs or rewrites. A D1 card's contract allows
operations, and the card is either autonomous or, in guided_preservation, a
proposal-bearing guided card. Leave every other card unchanged.

Every D1 card names one target and one concern. D1 asks whether the concern
change is **required**:
- **required:** the target's current form is wrong or cannot be assessed as it
  stands;
- **optional:** it could be more precise, finer, or organized differently.
  Optional refinements are not a reason to act.

Put valid optional refinements in `acceptable_alternatives`. Say in the
no_change_reason or the assessments why no required change remains.

## Variants

- **P, post-fix:** the parent's reference patch is already present in the
  variant workspace. `hypomap.grading.apply_proposed_updates` may draft it.
  Match provenance: cite real source spans whenever the parent's comparable
  objects do, and use `inferred` only where those objects are inferred too.
  Never use `proposed`.
  - Keep every public parent ID unchanged, including gaps. Never create a new
    public ID gap. A new object takes the next ID after the highest ID of its
    kind in the variant's public packet, with no suffix. If a hidden object
    holds that ID, move the hidden object to the next unused ID in the
    variant's files only (spec §4).
  - The reference is no-act on the named concern, with full assessments.
  - Forbidden inferences include "a change to the named target is required".
  - Discard the variant only if a *required* change remains on the named
    target.
- **I, inject:**
  - **evidence_audit:** plant exactly one defect on the named node that the
    visible evidence refutes. Prefer a reviewed forbidden inference made into
    an overclaim, or a mislink. Repair it with `REFINE_ASPECT` or
    `RELINK_EVIDENCE`.
  - **guided_preservation:** keep the proposal text unchanged and adjust the
    workspace so that the same proposal becomes justified. The reference
    accepts it.
  - In both, "no change is required" goes in the forbidden inferences.
- **Same text.** The variant uses exactly its parent's question, guidance and
  contract.
- **Never force a decision.** Discard a variant whose intended outcome is not
  clearly warranted from the public packet, and give the reason.

## Parent questions (question only)

For each D1 parent in the balanced families, write one named-concern question in
`question-rewrites.json` as {card_id, question}. Do not write the necessity
clause; code appends it.
- Do not edit the parent's seed, workspace, IDs, reference or demonstration.
- Findings about the existing parent packet are not yours to fix.
- For an unpaired no-act evidence_audit parent, name a node that the reference
  already assesses as supported.

## Delivery

Use the existing compile and check commands. Freeze public.json before writing
an answer. Public text states scientific premises only, never construction,
repair status or a verdict. Private metadata records parent_task_id, the method
(P or I), the named concern and the intended outcome.

Deliver the bundle, `question-rewrites.json` and `variant-decisions.json`,
including discards with reasons. There is one correction round.

## Opened assessment-only parents (spec §10)

Code opens the contract of an assessment-only evidence_audit parent. Write its
named-concern question as for any parent, naming the target's ID first. Then
write the reference by the necessity standard on that target, preferring a node
whose outcome the existing assessments already support:
- sound → no-act: keep the full assessments, add a no_change_reason, add "a
  change to the named target is required" to the forbidden inferences, and
  revise the demonstration;
- wrong or not assessable → act: add the minimal REFINE_ASPECT or
  RELINK_EVIDENCE on the target, and revise the requirements and
  demonstration.

Never edit the workspace. These parents are the only exception to "question
only".
