# Decision-variant Reviewer addendum

Read this after `paper-reviewer.md`. `docs/specs/decision-variants.md` §2–4
defines the design. You review one copied paper independently of its Author,
using gpt-6.1-sol / high.

Read whatever source the judgment needs. Do not re-audit unchanged content;
carry it forward.

## Necessity standard

For every D1 card, decide whether the concern change on the named target is
**required**: the target's current form is wrong or cannot be assessed as it
stands.
- A change that only makes the target more precise, finer or differently
  organized is **optional**. Optional changes do not make a no-act reference
  wrong. Record them as acceptable_alternatives.
- Use the spec §2 "required when" column for each family.

## Variants

- **Same text.** Parent and variant share the same question, guidance and
  contract text. The question names one target and one concern and does not
  reveal the outcome. For guided_preservation, the proposal is unchanged and
  only the workspace differs.
- **P:** the parent's patch is correctly represented, and no *required* change
  remains on the named target.
- **I:** exactly one refutable defect sits on the named node and is repaired by
  the concern operation. For guided_preservation, the proposal is justified.
- **IDs and provenance:**
  - public parent IDs are unchanged;
  - new objects take the next public IDs, with no suffixes;
  - fields and provenance match the parent's comparable objects: sourced where
    they are sourced, not defaulted to `inferred`, and never `proposed`;
  - no new public ID gaps, without exception (spec §4);
  - nothing public marks construction.
- **Discards.** Discard a variant whose intended outcome is not clearly
  warranted. Never relabel it to meet a quota.

## Parents

Re-confirm each parent reference under the necessity standard and the rewritten
question. Where the question admits other valid answers, record concrete
acceptable_alternatives. If the reference no longer holds, require the original
question to be restored and the card excluded from D1.

Record, but do not block on, findings about the existing parent packet.

Write one row per parent in `neutral-review-rN.json` with:
- `reference_valid`;
- `acceptable_alternatives`;
- `restore_original_question`;
- `include_d1_metrics`;
- `reasons`.

## Blind adjudication

Adjudicate each untouched first blind answer against its exact public packet.
Score D1 on the named concern only.

| Label | Meaning |
| --- | --- |
| agree | the blind D1 equals the reference D1 |
| required_act | the blind acts and the act is required, so the reference is wrong (task defect) |
| optional_act | the blind acts but the act is only optional (solver over-edit, not a task defect) |
| solver_error | any other wrong answer |

Record the blind D1, the reference D1, the label, and a one-sentence reason.
Never request a retry. Model review is not human review.

## Opened assessment-only parents (spec §10)

Review the reference as well as the question. The label follows the necessity
standard on the named target. For act, the patch is the minimal concern
operation and the requirements and demonstration match it. For no-act, the
no_change_reason and the forbidden inference are present. The workspace is
unchanged. If neither outcome is clearly warranted, exclude the card from D1.
