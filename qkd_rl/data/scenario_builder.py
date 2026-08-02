from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from qkd_rl.core.types import Edge, H5_LINK_TYPE_MAP, LinkType, Node, NodeType


@dataclass
class Scenario:
    nodes: list[Node]
    edges: list[Edge]
    start_t: int
    end_t: int
    slot_seconds: int

    @property
    def node_ids(self) -> list[str]:
        return [node.node_id for node in self.nodes]

    @property
    def edge_ids(self) -> list[str]:
        return [edge.edge_id for edge in self.edges]

    def edge_by_id(self) -> dict[str, Edge]:
        return {edge.edge_id: edge for edge in self.edges}


_NODE_TYPE_MAP = {
    "GS": NodeType.GS,
    "HAP": NodeType.HAP,
    "SAT": NodeType.SAT,
}


class ScenarioBuilder:
    def __init__(self, config: dict):
        self.config = config

    def build_small(self) -> Scenario:
        scenario_cfg = self.config["scenario"]
        rate_time_cfg = self.config["rate_provider"]["time"]
        nodes: list[Node] = []
        for idx in range(int(scenario_cfg["num_gs"])):
            nodes.append(Node(f"GS_{idx + 1:03d}", NodeType.GS))
        for idx in range(int(scenario_cfg["num_hap"])):
            nodes.append(Node(f"HAP_{idx + 1:03d}", NodeType.HAP))
        for idx in range(int(scenario_cfg["num_sat"])):
            nodes.append(Node(f"SAT_{idx + 1:03d}", NodeType.SAT))

        allowed = {LinkType(value) for value in scenario_cfg["allowed_link_types"]}
        edges: list[Edge] = []
        for i, src in enumerate(nodes):
            for dst in nodes[i + 1 :]:
                link_type = infer_link_type(src.node_type, dst.node_type)
                if link_type is None or link_type not in allowed:
                    continue
                edge_id = f"E_{src.node_id}__{dst.node_id}"
                edges.append(Edge(edge_id=edge_id, src=src.node_id, dst=dst.node_id, link_type=link_type))

        return Scenario(
            nodes=nodes,
            edges=edges,
            start_t=int(rate_time_cfg["start_index"]),
            end_t=int(rate_time_cfg["end_index"]),
            slot_seconds=int(rate_time_cfg["slot_seconds"]),
        )

    def build_full(self) -> Scenario:
        """Build the full-scale scenario from the H5 dataset registries.

        Node ids come from ``node_registry.csv``; candidate links come from
        ``link_registry.csv`` (falling back to the ``link_registry`` dataset
        inside ``link_data.h5``). This keeps the scenario and the
        ``H5RateProvider`` consistent with the same registry.
        """
        rate_time_cfg = self.config["rate_provider"]["time"]
        h5_cfg = self.config["rate_provider"].get("h5", {})
        dataset_dir = Path(h5_cfg.get("dataset_dir", "dataset/global"))
        node_registry_path = Path(h5_cfg.get("node_registry_path") or dataset_dir / "node_registry.csv")
        link_registry_path = Path(h5_cfg.get("link_registry_path") or dataset_dir / "link_registry.csv")
        link_data_path = Path(h5_cfg.get("link_data_path") or dataset_dir / "link_data.h5")

        if not node_registry_path.exists():
            raise FileNotFoundError(f"node_registry.csv not found: {node_registry_path}")

        node_rows = []
        with node_registry_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                node_rows.append(row)
        node_rows.sort(key=lambda row: int(row["node_id"]))

        nodes: list[Node] = []
        for row in node_rows:
            node_type = _NODE_TYPE_MAP[row["type"].strip().upper()]
            alt_km = float(row.get("alt_km") or 0.0)
            nodes.append(
                Node(
                    node_id=row["name"].strip(),
                    node_type=node_type,
                    lat=float(row.get("lat") or 0.0),
                    lon=float(row.get("lon") or 0.0),
                    alt_m=alt_km * 1000.0,
                )
            )

        link_rows: list[tuple[int, int, str]] = []
        if link_registry_path.exists():
            with link_registry_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    link_rows.append((int(row["node_u"]), int(row["node_v"]), row["link_type"].strip().upper()))
        else:
            import h5py

            with h5py.File(link_data_path, "r") as f:
                for item in f["link_registry"][:]:
                    link_rows.append(
                        (
                            int(item["node_u"]),
                            int(item["node_v"]),
                            bytes(item["link_type"]).decode("utf-8").strip().upper(),
                        )
                    )

        edges: list[Edge] = []
        for u, v, link_type_name in link_rows:
            link_type = H5_LINK_TYPE_MAP.get(link_type_name)
            if link_type is None:
                raise ValueError(f"Unknown H5 link type {link_type_name!r}")
            src = nodes[u].node_id
            dst = nodes[v].node_id
            edges.append(Edge(edge_id=f"E_{src}__{dst}", src=src, dst=dst, link_type=link_type))

        return Scenario(
            nodes=nodes,
            edges=edges,
            start_t=int(rate_time_cfg["start_index"]),
            end_t=int(rate_time_cfg["end_index"]),
            slot_seconds=int(rate_time_cfg["slot_seconds"]),
        )


def infer_link_type(a: NodeType, b: NodeType) -> LinkType | None:
    types = {a, b}
    if types == {NodeType.GS, NodeType.HAP}:
        return LinkType.GS_HAP
    if types == {NodeType.GS, NodeType.SAT}:
        return LinkType.GS_SAT
    if types == {NodeType.HAP, NodeType.SAT}:
        return LinkType.HAP_SAT
    if a == NodeType.SAT and b == NodeType.SAT:
        return LinkType.SAT_SAT
    return None