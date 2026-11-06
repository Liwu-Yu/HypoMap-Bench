# Leak checker

Check ONE paper's cards before review for public text that gives away the
answer. Read only the public.json and answer.json paths you are given. Do not
open the source, workspace.json, seeds, reviews, other papers, or earlier
collections. Do not delegate.

For each card:
1. From answer.json, name the reference's central decision in one line (the
   gap, the duplicate, the flaw, that no change is needed, the alternative).
2. Read every public field except question and user_guidance: objective,
   background, map nodes and edges, experiment plans, observations,
   interpretations, constraints, and every inference_reason, scope,
   limitation and note.
3. Flag text that states or strongly implies that decision, or that reveals
   how the card was built. Examples from earlier batches: "no hardness
   measurement" on a card whose answer is the missing hardness aspect; "is
   re-quantified" or "restates ... that this map already covers" on a
   redundancy card; "scoped restatement" or "correctly scoped" on a no-change
   control; "constructed", "for this task", "control wording".
4. Flag ID gaps: if a card view skips an ID (for example n02, then n04) where
   the withheld target of the decision would sit, list it as a leak with
   field "ids", quote the IDs, and give the renumbering as neutral_rewrite.
   Report it in leak-check.json, not only in your summary.
5. Do not flag facts the student must reason from: what was measured, in whom,
   with which comparison, which data or samples an analysis used, and what the
   source reports. Inferring the gap or the duplicate from those facts is the
   student's job.

Save leak-check.json in the paper folder:
{"paper_id": "...", "cards": [{"card_id": "c01", "central_decision": "...",
  "leaks": [{"object_id": "...", "field": "...", "quote": "...", "why": "...",
  "neutral_rewrite": "..."}]}]}
Use an empty leaks list when a card has none. Finish with the number of cards
that have leaks.
