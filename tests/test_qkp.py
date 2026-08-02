from qkd_rl.core.types import Edge, LinkType
from qkd_rl.env.qkp import LinkQKPPool


def test_link_qkp_capacity_and_path_consume():
    edges = [
        Edge("e1", "a", "b", LinkType.GS_HAP),
        Edge("e2", "b", "c", LinkType.HAP_SAT),
    ]
    qkp = LinkQKPPool(
        edges,
        {
            "initial_level": 0.0,
            "key_ttl_steps": 10,
            "capacity": {
                "default": 10.0,
                "per_link_type": {"gs_hap": 20.0},
                "per_edge": {"e2": 30.0},
            },
        },
    )
    qkp.reset()
    assert qkp.get_capacity("e1") == 20.0
    assert qkp.get_capacity("e2") == 30.0
    assert qkp.add_keys("e1", 25.0, 0) == 20.0
    assert qkp.add_keys("e2", 5.0, 0) == 5.0
    assert not qkp.consume_path(["e1", "e2"], 6.0)
    assert qkp.get_level("e1") == 20.0
    assert qkp.consume_path(["e1", "e2"], 5.0)
    assert qkp.get_level("e1") == 15.0
    assert qkp.get_level("e2") == 0.0

