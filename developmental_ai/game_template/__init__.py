"""Game onboarding template (WP-16, ARC-005).

Everything a new game needs to join SkyBot's ONE universal brain lives in this
package's pattern: a game interface (adapter, launcher, install step, game
config) plus, separately, an evaluator-side checkpoint detector in
`developmental_ai.evaluator`. No learning module changes to add a game.

Nothing here is imported by learning code; a game's YAML names its adapter
(`backend: plugin`, `adapter: "pkg.module:Class"`) and `make_env` imports it
lazily. Status: built, untested (GOV-011 entry WP-16).
"""
