"""
asset_graph.py — Critical-path Business Interruption module
============================================================
Replaces the scalar sector-phi × ELF approximation with a topology-aware
expected downtime calculation derived from the asset's component dependency
graph (DAG).

Why this matters
----------------
Two assets in the same sector can have wildly different BI exposure:
  • A chemical plant with a single 66kV substation feeding all process utilities
    → substation failure takes the whole site down for 14 days (long recovery chain)
  • An identical plant with dual-feed HV supply + diesel backup
    → the same flood event causes ≤3 days downtime before backup kicks in

The old phi*ELF model can't see the difference.  This module computes:

    downtime_fraction = P(system_is_down | hazard_probs)
                        × max_repair_path_days / 365

using a two-pass DAG traversal:
  Pass 1 (forward)  — propagate failure probabilities from sources to sinks
  Pass 2 (backward) — propagate max repair-path lengths from sinks to sources
Then combine: E[downtime_days] = P(sink_is_down) × max_repair_chain[sink]

Sources
-------
Failure probability model calibrated against:
  • Willis Towers Watson "Climate Physical Risk: Asset-Level Sensitivity" (2023)
  • Swiss Re Sigma No.4/2022 — industrial downtime loss ratios
  • IPCC AR6 Ch11 Table 11.9 — infrastructure failure probabilities by hazard
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class ComponentNode:
    """
    One physical component in an asset's operational dependency chain.

    Parameters
    ----------
    name : str
        Unique identifier (e.g. "66kV_substation", "reactor_A").
    hazard_sensitivities : dict
        Maps hazard name → sensitivity scalar ∈ [0, 1].
        Interpreted as: at full hazard probability (p=1.0), this component
        fails with probability = sensitivity.  Combined as:
            P(self_fails) = 1 - ∏(1 - p_h × s_h)
    recovery_days : int
        Expected days to restore this component after failure.
        Used to compute the critical repair path.
    has_redundancy : bool
        If True, the component has a backup (e.g. dual-feed power, diesel
        generator).  Failure probability is squared — both primary AND backup
        must fail simultaneously.
    """
    name: str
    hazard_sensitivities: Dict[str, float] = field(default_factory=dict)
    recovery_days: int = 7
    has_redundancy: bool = False


@dataclass
class ComponentGraph:
    """
    DAG of component dependencies for one asset.

    Attributes
    ----------
    nodes : dict
        name → ComponentNode
    upstream : dict
        name → list of node names that this node depends on.
        If any upstream is down, this node is also down (AND-logic).
        Source nodes (no upstream) fail only from their own hazard exposure.
    description : str
        Human-readable label used in the audit trail.
    """
    nodes: Dict[str, ComponentNode] = field(default_factory=dict)
    upstream: Dict[str, List[str]] = field(default_factory=dict)
    description: str = ""

    def add_node(self, node: ComponentNode, depends_on: Optional[List[str]] = None) -> None:
        self.nodes[node.name] = node
        self.upstream[node.name] = depends_on or []

    def sink_nodes(self) -> List[str]:
        """Nodes that no other node depends on (outputs of the system)."""
        all_upstream = {u for ups in self.upstream.values() for u in ups}
        return [n for n in self.nodes if n not in all_upstream]


# ── Core algorithm ───────────────────────────────────────────────────────────

def _topological_order(graph: ComponentGraph) -> List[str]:
    """Kahn's algorithm — returns processing order (sources first)."""
    in_deg: Dict[str, int] = {n: len(graph.upstream.get(n, [])) for n in graph.nodes}
    queue: deque = deque(n for n, d in in_deg.items() if d == 0)
    order: List[str] = []

    # Build reverse adjacency (who depends on me?)
    dependents: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    for node, ups in graph.upstream.items():
        for u in ups:
            dependents[u].append(node)

    while queue:
        n = queue.popleft()
        order.append(n)
        for dep in dependents[n]:
            in_deg[dep] -= 1
            if in_deg[dep] == 0:
                queue.append(dep)

    if len(order) != len(graph.nodes):
        raise ValueError("ComponentGraph contains a cycle — invalid DAG.")
    return order


def critical_path_downtime(
    graph: ComponentGraph,
    hazard_probs: Dict[str, float],
) -> Tuple[float, Dict]:
    """
    Compute expected annual downtime fraction for the asset given current
    hazard probabilities.

    Returns
    -------
    downtime_fraction : float
        Expected fraction of the year the system output is unavailable.
        Replaces phi × ELF in the BI revenue-at-risk calculation.
    audit : dict
        Step-by-step arithmetic for regulatory transparency.
    """
    order = _topological_order(graph)

    # Build reverse adjacency for repair-path pass
    dependents: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    for node, ups in graph.upstream.items():
        for u in ups:
            dependents[u].append(node)

    # ── Pass 1: forward — failure probability propagation ───────────────────
    p_fail: Dict[str, float] = {}
    for name in order:
        node = graph.nodes[name]

        # Own failure from direct hazard exposure
        p_work_own = 1.0
        for hazard, sens in node.hazard_sensitivities.items():
            p_h = hazard_probs.get(hazard, 0.0)
            p_work_own *= (1.0 - p_h * sens)
        p_fail_own = 1.0 - p_work_own
        if node.has_redundancy:
            p_fail_own = p_fail_own ** 2  # backup must also fail

        # Cascade from upstream: node is down if ANY upstream is down
        # (AND-logic on inputs: all inputs must be working)
        p_all_upstream_work = 1.0
        for u in graph.upstream.get(name, []):
            p_all_upstream_work *= (1.0 - p_fail.get(u, 0.0))

        # Effective: node works only if it doesn't fail itself AND all upstream work
        p_work_effective = (1.0 - p_fail_own) * p_all_upstream_work
        p_fail[name] = max(0.0, min(1.0, 1.0 - p_work_effective))

    # ── Pass 2: backward — max repair-chain length ──────────────────────────
    # max_repair[node] = recovery_days[node] + max(max_repair[upstream_i])
    # This is the critical sequential repair path: you can't fix downstream
    # until upstream is restored first.
    max_repair: Dict[str, float] = {}
    for name in order:
        own_days = graph.nodes[name].recovery_days
        if graph.upstream.get(name):
            upstream_max = max(max_repair.get(u, 0.0) for u in graph.upstream[name])
        else:
            upstream_max = 0.0
        max_repair[name] = own_days + upstream_max

    # ── Combine at sink nodes ────────────────────────────────────────────────
    sinks = graph.sink_nodes()
    if not sinks:
        sinks = [order[-1]]  # fallback: last node in topological order

    # System expected downtime = worst-case sink
    worst_fraction = 0.0
    worst_sink = sinks[0]
    for sink in sinks:
        fraction = p_fail[sink] * (max_repair[sink] / 365.0)
        if fraction > worst_fraction:
            worst_fraction = fraction
            worst_sink = sink

    # Build audit trail
    audit = {
        "model": "Critical-path DAG (ClimRisk asset_graph.py)",
        "description": graph.description,
        "hazard_probs_used": {k: round(v, 4) for k, v in hazard_probs.items()},
        "node_failure_probs": {k: round(v, 4) for k, v in p_fail.items()},
        "node_max_repair_days": {k: round(v, 1) for k, v in max_repair.items()},
        "sink_nodes": sinks,
        "critical_sink": worst_sink,
        "p_system_down": round(p_fail.get(worst_sink, 0.0), 4),
        "max_repair_chain_days": round(max_repair.get(worst_sink, 0.0), 1),
        "downtime_fraction": round(worst_fraction, 5),
    }

    return worst_fraction, audit


# ── Sector default graphs ────────────────────────────────────────────────────
# Pre-built for clients without explicit component manifests.
# Calibrated against Swiss Re industrial loss data and IEA infrastructure
# failure statistics.  Clients with detailed asset registers can pass their
# own ComponentGraph to assess_physical_risk_financial().

def _power_generation_graph() -> ComponentGraph:
    """Coal / gas / thermal power plant dependency chain."""
    g = ComponentGraph(description="Thermal power generation — single-train")
    g.add_node(ComponentNode("fuel_supply",
        hazard_sensitivities={"flood": 0.55, "wildfire": 0.40, "water_stress": 0.30},
        recovery_days=5))
    g.add_node(ComponentNode("water_intake",
        hazard_sensitivities={"flood": 0.35, "water_stress": 0.80, "heat_stress": 0.20},
        recovery_days=10))
    g.add_node(ComponentNode("cooling_system",
        hazard_sensitivities={"heat_stress": 0.45, "water_stress": 0.60, "flood": 0.25},
        recovery_days=14),
        depends_on=["water_intake"])
    g.add_node(ComponentNode("boiler_steam_gen",
        hazard_sensitivities={"flood": 0.50, "heat_stress": 0.20},
        recovery_days=30),
        depends_on=["fuel_supply", "cooling_system"])
    g.add_node(ComponentNode("turbine_generator",
        hazard_sensitivities={"flood": 0.30, "wind": 0.20},
        recovery_days=45),
        depends_on=["boiler_steam_gen"])
    g.add_node(ComponentNode("hv_switchgear",
        hazard_sensitivities={"flood": 0.45, "wind": 0.50},
        recovery_days=7),
        depends_on=["turbine_generator"])
    g.add_node(ComponentNode("grid_connection",
        hazard_sensitivities={"wind": 0.60, "flood": 0.30},
        recovery_days=3,
        has_redundancy=False),
        depends_on=["hv_switchgear"])
    return g


def _chemicals_refining_graph() -> ComponentGraph:
    """Chemical plant / refinery with single process train."""
    g = ComponentGraph(description="Chemicals / refining — single-train process")
    g.add_node(ComponentNode("grid_power",
        hazard_sensitivities={"flood": 0.40, "wind": 0.55},
        recovery_days=3,
        has_redundancy=False))
    g.add_node(ComponentNode("water_supply",
        hazard_sensitivities={"water_stress": 0.75, "flood": 0.35},
        recovery_days=7))
    g.add_node(ComponentNode("process_utilities",
        hazard_sensitivities={"heat_stress": 0.30, "flood": 0.40},
        recovery_days=10),
        depends_on=["grid_power", "water_supply"])
    g.add_node(ComponentNode("reactor_distillation",
        hazard_sensitivities={"heat_stress": 0.50, "flood": 0.60, "wind": 0.25},
        recovery_days=60),
        depends_on=["process_utilities"])
    g.add_node(ComponentNode("separation_recovery",
        hazard_sensitivities={"flood": 0.35},
        recovery_days=21),
        depends_on=["reactor_distillation"])
    g.add_node(ComponentNode("storage_dispatch",
        hazard_sensitivities={"flood": 0.50, "wind": 0.30},
        recovery_days=14),
        depends_on=["separation_recovery"])
    return g


def _mining_graph() -> ComponentGraph:
    """Open-cut or underground mine with processing and export chain."""
    g = ComponentGraph(description="Mining — extraction to export terminal")
    g.add_node(ComponentNode("site_power",
        hazard_sensitivities={"flood": 0.35, "wind": 0.45, "wildfire": 0.30},
        recovery_days=4,
        has_redundancy=False))
    g.add_node(ComponentNode("dewatering_pumps",
        hazard_sensitivities={"flood": 0.85, "water_stress": 0.15},
        recovery_days=10),
        depends_on=["site_power"])
    g.add_node(ComponentNode("extraction_equipment",
        hazard_sensitivities={"heat_stress": 0.40, "wildfire": 0.35, "flood": 0.30},
        recovery_days=21),
        depends_on=["site_power", "dewatering_pumps"])
    g.add_node(ComponentNode("primary_crushing",
        hazard_sensitivities={"flood": 0.35, "heat_stress": 0.20},
        recovery_days=14),
        depends_on=["extraction_equipment"])
    g.add_node(ComponentNode("processing_plant",
        hazard_sensitivities={"flood": 0.40, "water_stress": 0.55, "heat_stress": 0.30},
        recovery_days=30),
        depends_on=["primary_crushing"])
    g.add_node(ComponentNode("conveyors_rail",
        hazard_sensitivities={"flood": 0.55, "wind": 0.35, "wildfire": 0.40},
        recovery_days=14),
        depends_on=["processing_plant"])
    g.add_node(ComponentNode("export_terminal",
        hazard_sensitivities={"flood": 0.45, "wind": 0.60},
        recovery_days=7),
        depends_on=["conveyors_rail"])
    return g


def _manufacturing_graph() -> ComponentGraph:
    """Discrete or continuous manufacturing facility."""
    g = ComponentGraph(description="Industrial manufacturing — production to dispatch")
    g.add_node(ComponentNode("grid_power",
        hazard_sensitivities={"flood": 0.35, "wind": 0.50},
        recovery_days=3,
        has_redundancy=False))
    g.add_node(ComponentNode("raw_material_intake",
        hazard_sensitivities={"flood": 0.45, "wildfire": 0.30, "wind": 0.25},
        recovery_days=7))
    g.add_node(ComponentNode("production_line",
        hazard_sensitivities={"heat_stress": 0.35, "flood": 0.40},
        recovery_days=14),
        depends_on=["grid_power", "raw_material_intake"])
    g.add_node(ComponentNode("finished_goods_warehouse",
        hazard_sensitivities={"flood": 0.55, "wind": 0.35},
        recovery_days=10),
        depends_on=["production_line"])
    g.add_node(ComponentNode("distribution_logistics",
        hazard_sensitivities={"flood": 0.40, "wind": 0.30, "wildfire": 0.25},
        recovery_days=5),
        depends_on=["finished_goods_warehouse"])
    return g


def _oil_gas_graph() -> ComponentGraph:
    """Upstream / midstream oil & gas: wellhead to terminal."""
    g = ComponentGraph(description="Oil & gas — wellhead to export / processing terminal")
    g.add_node(ComponentNode("wellhead_infrastructure",
        hazard_sensitivities={"flood": 0.30, "wind": 0.40, "heat_stress": 0.20},
        recovery_days=14))
    g.add_node(ComponentNode("processing_facilities",
        hazard_sensitivities={"flood": 0.45, "heat_stress": 0.35, "wind": 0.30},
        recovery_days=45),
        depends_on=["wellhead_infrastructure"])
    g.add_node(ComponentNode("pipeline_network",
        hazard_sensitivities={"flood": 0.35, "wildfire": 0.30, "wind": 0.25},
        recovery_days=21),
        depends_on=["processing_facilities"])
    g.add_node(ComponentNode("storage_terminal",
        hazard_sensitivities={"flood": 0.50, "wind": 0.55, "heat_stress": 0.25},
        recovery_days=10),
        depends_on=["pipeline_network"])
    return g


def _real_estate_graph() -> ComponentGraph:
    """Commercial or industrial real estate — building systems."""
    g = ComponentGraph(description="Commercial real estate — building systems")
    g.add_node(ComponentNode("structural_envelope",
        hazard_sensitivities={"flood": 0.40, "wind": 0.55, "heat_stress": 0.15},
        recovery_days=90))
    g.add_node(ComponentNode("electrical_systems",
        hazard_sensitivities={"flood": 0.60, "wind": 0.35},
        recovery_days=14),
        depends_on=["structural_envelope"])
    g.add_node(ComponentNode("hvac_cooling",
        hazard_sensitivities={"heat_stress": 0.60, "flood": 0.30, "water_stress": 0.40},
        recovery_days=14),
        depends_on=["electrical_systems"])
    g.add_node(ComponentNode("access_transport",
        hazard_sensitivities={"flood": 0.55, "wildfire": 0.35, "wind": 0.30},
        recovery_days=7))
    g.add_node(ComponentNode("building_operations",
        hazard_sensitivities={"heat_stress": 0.20},
        recovery_days=3),
        depends_on=["structural_envelope", "electrical_systems", "hvac_cooling", "access_transport"])
    return g


def _agriculture_food_graph() -> ComponentGraph:
    """Agricultural production or food processing."""
    g = ComponentGraph(description="Agriculture / food processing — production to cold chain")
    g.add_node(ComponentNode("irrigation_water",
        hazard_sensitivities={"water_stress": 0.85, "flood": 0.30},
        recovery_days=14))
    g.add_node(ComponentNode("field_production",
        hazard_sensitivities={"heat_stress": 0.70, "flood": 0.55, "wildfire": 0.45,
                               "water_stress": 0.60},
        recovery_days=180),
        depends_on=["irrigation_water"])  # crop loss — season length
    g.add_node(ComponentNode("cold_storage",
        hazard_sensitivities={"flood": 0.40, "heat_stress": 0.35},
        recovery_days=14))
    g.add_node(ComponentNode("processing_plant",
        hazard_sensitivities={"flood": 0.35, "heat_stress": 0.25, "water_stress": 0.30},
        recovery_days=21),
        depends_on=["field_production", "cold_storage"])
    g.add_node(ComponentNode("distribution",
        hazard_sensitivities={"flood": 0.40, "wind": 0.30},
        recovery_days=5),
        depends_on=["processing_plant"])
    return g


def _transport_infrastructure_graph() -> ComponentGraph:
    """Port, rail, or road transport hub."""
    g = ComponentGraph(description="Transport / logistics hub")
    g.add_node(ComponentNode("civil_infrastructure",
        hazard_sensitivities={"flood": 0.65, "wind": 0.50, "wildfire": 0.20},
        recovery_days=30))
    g.add_node(ComponentNode("power_signalling",
        hazard_sensitivities={"flood": 0.45, "wind": 0.55},
        recovery_days=7),
        depends_on=["civil_infrastructure"])
    g.add_node(ComponentNode("cargo_handling",
        hazard_sensitivities={"wind": 0.60, "flood": 0.50, "heat_stress": 0.25},
        recovery_days=10),
        depends_on=["power_signalling"])
    g.add_node(ComponentNode("access_links",
        hazard_sensitivities={"flood": 0.55, "wildfire": 0.35, "wind": 0.40},
        recovery_days=14))
    g.add_node(ComponentNode("operational_output",
        hazard_sensitivities={},
        recovery_days=2),
        depends_on=["cargo_handling", "access_links"])
    return g


# ── Sector dispatch table ────────────────────────────────────────────────────

_SECTOR_GRAPHS: Dict[str, ComponentGraph] = {
    "utilities":        _power_generation_graph(),
    "power":            _power_generation_graph(),
    "energy":           _power_generation_graph(),
    "chemicals":        _chemicals_refining_graph(),
    "oil_gas":          _chemicals_refining_graph(),
    "refining":         _chemicals_refining_graph(),
    "mining":           _mining_graph(),
    "metals_mining":    _mining_graph(),
    "coal":             _mining_graph(),
    "industrials":      _manufacturing_graph(),
    "manufacturing":    _manufacturing_graph(),
    "automotive":       _manufacturing_graph(),
    "cement":           _manufacturing_graph(),
    "steel":            _manufacturing_graph(),
    "real_estate":      _real_estate_graph(),
    "construction":     _real_estate_graph(),
    "agriculture":      _agriculture_food_graph(),
    "food_beverage":    _agriculture_food_graph(),
    "beverages":        _agriculture_food_graph(),
    "consumer_staples": _agriculture_food_graph(),
    "transport":        _transport_infrastructure_graph(),
    "shipping":         _transport_infrastructure_graph(),
    "aviation":         _transport_infrastructure_graph(),
    "logistics":        _transport_infrastructure_graph(),
}


def get_sector_graph(sector: str) -> Optional[ComponentGraph]:
    """
    Return the pre-built ComponentGraph for a sector, or None if not mapped.

    Parameters
    ----------
    sector : str
        Free-text sector — matched case-insensitively.
    """
    key = sector.lower().replace(" ", "_").replace("-", "_")
    return _SECTOR_GRAPHS.get(key)
