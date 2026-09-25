# v4 model contract

- **Observation graph:** current legal rate-bearing physical links **union** links
  with positive QKP inventory, even when their current rate is zero. Only the
  original legal rate-bearing arcs are passed to the unchanged matching policy.
- **Pair mask:** for each ground-station pair, use the union of *all simple*
  source-to-destination paths of at most 4 physical hops on this observed graph.
  A bounded reverse BFS prunes a local DFS; disconnected pairs have empty masks.
- **Isolation:** the node/physical-edge features do not contain aggregate demand
  features, and the GNN only receives physical edges. A demand pair token comes
  from its own raw request statistics and the two GS embeddings after physical
  encoding; other pair tokens never enter this GNN.
- **Attention:** link self-attention runs separately on each pair's BFS-filtered
  physical links. Candidate directed arcs attend only to those same links with
  that pair's demand token. Weights are shared across pairs, **activations are
  not**. The existing weighted merge produces one edge score for the unchanged
  joint matching sampler. Legal arcs outside every pair path retain the
  preparation fallback, which uses pooled per-pair context but no global
  self-attention. The critic remains global by design.
- **Experiment:** `scripts/experiment.py train --model v4` (8 BC episodes,
  shared short-run config with 8 x 240 rollout steps per update); evaluate
  using the unmodified
  `scripts/eval/frozen_eval.py` (windows A/B). The v3 implementation was copied
  into this directory rather than imported so the v4 run manifest fingerprints
  its complete source. Treat all pre-v4 checkpoints as incompatible.
