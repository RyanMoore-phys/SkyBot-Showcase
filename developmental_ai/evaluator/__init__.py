"""EVALUATOR-SIDE package (ADR-002, WP-16, WP-14). Learning code NEVER imports this.

Checkpoint detectors read the RED channel (`info["evaluator"]`, engine counters
and events) and say which checkpoint a run has reached. They are evaluation
only: no reward, no learning input, no target (ASS-007 section 2). The rule is
enforced mechanically by tests/_evaluator_isolation_smoke.py: no module under
developmental_ai/ outside this package may import it.

Importers allowed: tools/ (evaluation machinery), tests/, and this package.
"""
