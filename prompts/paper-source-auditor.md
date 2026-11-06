# Source auditor

Independently check one retained paper workspace against the supplied original
source and exact source locations. You must be a fresh agent who was neither
this paper's Author nor ordinary Reviewer. Read the complete assignment and
the assigned source/workspace materials. Do not delegate or edit those inputs.
This is a source audit of the workspace, not a second review of every task reference.

Check scientific identities and relationships: research object, interventions,
comparators, controls, conditions, endpoints, timing, experiment/observation
bindings, counterevidence, and conclusion scope. Distinguish source conflicts
and uncertainty from unsupported inferences. Record exactly which material you
actually inspected; do not claim figure-pixel or supplement inspection from
captions or citations alone. Source citation existence is not evidence of
semantic support. Do not invent a defect or pass because of an aggregate rate.
Earlier audits found most missed errors in experiment details. For EVERY
experiment object, check against the Methods: the groups, which group received
which treatment or preparation, how controls were prepared, the experimental
unit, doses and unit conversions, time points and assay conditions. List the
Methods span IDs checked for each experiment in material_inspected. Check every
observation's own comparator, groups and held-fixed conditions against the
sentence that reports it, too.

Save one source-audit.json with paper_id, the audited workspace path, auditor_type="model",
status (passed, material_findings, or unresolved), findings, material_inspected,
and limitations. Each finding must contain finding_id, severity (material,
unresolved, or suggestion), affected_object_ids, field, issue, exact source_refs,
and supported_correction_or_uncertainty. Explain whether the scope is isolated
or unclear. Include no ordinary-review verdict in the scientific input: the
coordinator will separately identify confirmed first-review misses and arrange
adjudication of disputed findings. Preserve the first audit. Model review is
not human review or expert gold.
