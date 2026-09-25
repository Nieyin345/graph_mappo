"""v5 marker -- the implementation lives on the server branch codex/pairhist-v4.

See README.md in this directory. This module exists so the model registry
(``qkd_rl.model_zoo.model_name``) and tooling can resolve ``model: v5`` to a
directory, and so an accidental ``--model v5`` run fails loudly with a pointer
to the branch instead of silently importing the wrong implementation.
"""

from __future__ import annotations

_BRANCH = "codex/pairhist-v4"


def build_model(action_space, config):
    raise NotImplementedError(
        "v5's implementation is not in this repository -- it lives on server "
        f"branch {_BRANCH!r}. This directory holds only its artifact snapshot "
        "under server_results/pairhist_v5/. Check out that branch to reproduce; "
        "the current worktree cannot load v5 checkpoints."
    )


def graph_builder_class():
    raise NotImplementedError(
        f"v5's graph builder lives on server branch {_BRANCH!r}, not here."
    )
