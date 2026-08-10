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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (for the smart active-node selection)."""
    import math

    r_earth = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r_earth * math.asin(math.sqrt(a))


def _sat_orbit_rank(name: str) -> int:
    """Prefer inclined orbits over equatorial/polar/GEO when picking satellites.

    0 = inclined (``Mid``/``MEO``), 1 = unknown/other, 2 = Eq/Pol/GEO.
    """
    lowered = name.lower()
    if "mid" in lowered or "meo" in lowered:
        return 0
    if "eq" in lowered or "pol" in lowered or "geo" in lowered:
        return 2
    return 1


class ScenarioBuilder:
    def __init__(self, config: dict):
        self.config = config

    def _apply_active_nodes(self, nodes: list[Node]) -> list[Node]:
        """Keep only the configured active-node subset (0 = keep all).

        Global knob shared by every algorithm (RL + all baselines):
        ``scenario.active_nodes`` caps the number of GS / HAP / SAT nodes kept
        and how they are picked, shrinking the graph, the action space and the
        request pairs for quick low-difficulty experiments.  A limit of 0 (or a
        missing section) keeps every node of that type.  Callers must also drop
        edges whose endpoints are removed.

        ``mode: first_n`` (default) keeps the first N nodes of each type in
        registry order.  ``mode: smart`` picks a geographically coherent subset:

        * GS: the N ground stations closest to ``gs_center`` (default: the
          centroid of all GS), so remote stations are avoided;
        * HAP: the N platforms closest to the already-selected GS set, but only
          platforms within ``hap_gs_max_km`` (default 700 km, matching the H5
          HAP-GS static distance prefilter) of some selected GS count - a HAP
          with no GS nearby is useless in the subgraph, so the count is not
          padded with it;
        * SAT: the N satellites with an inclined orbit (``Mid``/``MEO``) first,
          skipping equatorial (``Eq``), polar (``Pol``) and GEO satellites when
          enough inclined ones exist.
        """
        cfg = self.config.get("scenario", {}).get("active_nodes", {}) or {}
        limits = {
            NodeType.GS: int(cfg.get("gs_count", 0) or 0),
            NodeType.HAP: int(cfg.get("hap_count", 0) or 0),
            NodeType.SAT: int(cfg.get("sat_count", 0) or 0),
        }
        if not any(limits.values()):
            return nodes
        mode = str(cfg.get("mode", "first_n") or "first_n")
        if mode == "first_n":
            return self._apply_first_n(nodes, limits)
        if mode == "smart":
            return self._apply_smart(nodes, limits, cfg)
        raise ValueError(f"Unknown active_nodes mode: {mode!r}")

    def _apply_first_n(self, nodes: list[Node], limits: dict[NodeType, int]) -> list[Node]:
        kept: list[Node] = []
        seen: dict[NodeType, int] = {}
        for node in nodes:
            limit = limits[node.node_type]
            if limit <= 0:
                kept.append(node)
                continue
            count = seen.get(node.node_type, 0)
            if count < limit:
                kept.append(node)
                seen[node.node_type] = count + 1
        return kept

    def _apply_smart(self, nodes: list[Node], limits: dict[NodeType, int], cfg: dict) -> list[Node]:
        gs = [node for node in nodes if node.node_type is NodeType.GS]
        hap = [node for node in nodes if node.node_type is NodeType.HAP]
        sat = [node for node in nodes if node.node_type is NodeType.SAT]

        n_gs = limits[NodeType.GS] or len(gs)
        n_hap = limits[NodeType.HAP] or len(hap)
        n_sat = limits[NodeType.SAT] or len(sat)

        def _lat(node: Node) -> float:
            return 0.0 if node.lat is None else float(node.lat)

        def _lon(node: Node) -> float:
            return 0.0 if node.lon is None else float(node.lon)

        center = cfg.get("gs_center")
        if not center:
            if gs:
                center = (sum(_lat(node) for node in gs) / len(gs), sum(_lon(node) for node in gs) / len(gs))
            else:
                center = (0.0, 0.0)
        selected_gs = sorted(gs, key=lambda node: _haversine_km(_lat(node), _lon(node), center[0], center[1]))[:n_gs]

        def _nearest_selected_km(node: Node) -> float:
            return min(
                (_haversine_km(_lat(node), _lon(node), _lat(g), _lon(g)) for g in selected_gs),
                default=0.0,
            )

        if limits[NodeType.HAP] > 0:
            hap_gs_max_km = float(cfg.get("hap_gs_max_km", 700.0) or 700.0)
            ranked_hap = sorted(((h, _nearest_selected_km(h)) for h in hap), key=lambda item: item[1])
            # Only platforms with a GS within range are useful (the H5 link set
            # itself only keeps HAP-GS pairs within this distance), so a far-away
            # HAP is dropped instead of padding the requested count.
            selected_hap = [h for h, dist_km in ranked_hap if dist_km <= hap_gs_max_km][:n_hap]
        else:
            selected_hap = hap
        selected_sat = sorted(sat, key=lambda node: (_sat_orbit_rank(node.node_id), sat.index(node)))[:n_sat]
        return selected_gs + selected_hap + selected_sat

    def _apply_time_limit(self, start_t: int, end_t: int) -> tuple[int, int]:
        """Cap the simulation horizon with the global scenario.time_limit knob.

        Same philosophy as active_nodes: one global setting shared by every
        algorithm (RL + all baselines) that shrinks the problem for quick
        experiments. days > 0 simulates only the first N days (of
        env.day_steps slots each) from start_t; days: 0 (default) keeps the
        full rate_provider.time window. The cap never extends the configured
        window.
        """
        time_cfg = self.config.get("scenario", {}).get("time_limit", {}) or {}
        days = int(time_cfg.get("days", 0) or 0)
        if days <= 0:
            return start_t, end_t
        day_steps = int(self.config.get("env", {}).get("day_steps", 1440) or 1440)
        capped_end = start_t + days * day_steps
        return start_t, min(end_t, capped_end)
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
        nodes = self._apply_active_nodes(nodes)

        allowed = {LinkType(value) for value in scenario_cfg["allowed_link_types"]}
        edges: list[Edge] = []
        for i, src in enumerate(nodes):
            for dst in nodes[i + 1 :]:
                link_type = infer_link_type(src.node_type, dst.node_type)
                if link_type is None or link_type not in allowed:
                    continue
                edge_id = f"E_{src.node_id}__{dst.node_id}"
                edges.append(Edge(edge_id=edge_id, src=src.node_id, dst=dst.node_id, link_type=link_type))

        start_t, end_t = self._apply_time_limit(
            int(rate_time_cfg["start_index"]), int(rate_time_cfg["end_index"])
        )
        return Scenario(
            nodes=nodes,
            edges=edges,
            start_t=start_t,
            end_t=end_t,
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

        all_nodes = nodes
        nodes = self._apply_active_nodes(nodes)
        kept_ids = {node.node_id for node in nodes}

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
            src = all_nodes[u].node_id
            dst = all_nodes[v].node_id
            if src not in kept_ids or dst not in kept_ids:
                continue  # active-node subset: drop edges to removed nodes
            edges.append(Edge(edge_id=f"E_{src}__{dst}", src=src, dst=dst, link_type=link_type))

        start_t, end_t = self._apply_time_limit(
            int(rate_time_cfg["start_index"]), int(rate_time_cfg["end_index"])
        )
        return Scenario(
            nodes=nodes,
            edges=edges,
            start_t=start_t,
            end_t=end_t,
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