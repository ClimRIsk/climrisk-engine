"""
ai/tools.py — ClimRisk LLM Tool Registry

Defines every engine capability as a Groq-compatible tool schema, paired
with a Python executor that calls the internal engine directly (no HTTP
round-trip, no auth overhead).

Each entry in TOOL_REGISTRY has:
  definition  — the JSON schema sent to the LLM in the tools= array
  executor    — a Python callable(args: dict) -> dict that runs the tool

Design rules
────────────
• All executors are synchronous and self-contained.
• Executors import lazily — the LLM agent only pays the import cost when
  a tool is actually called.
• Every executor returns a plain dict.  The LLM agent serialises it to
  JSON and injects it as the tool_result message.
• Errors are caught and returned as {"error": "..."}  so the LLM can
  reason about failures rather than crashing.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────────
ToolExecutor = Callable[[dict], dict]


# ─────────────────────────────────────────────────────────────────────────────
# 1. assess_company
# ─────────────────────────────────────────────────────────────────────────────

_ASSESS_DEFINITION = {
    "type": "function",
    "function": {
        "name": "assess_company",
        "description": (
            "Run a full parametric climate risk assessment for any company. "
            "Returns physical risk scores per NGFS scenario (NZE / Delayed / Current Policies), "
            "transition risk, Monte Carlo Climate VaR, ND-GAIN country vulnerability, "
            "and a 2025-2050 trajectory. Works for any company in any country without "
            "prior data — just needs sector and jurisdiction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Name of the company to assess (e.g. 'Shell', 'HDFC Bank', 'Toyota')"
                },
                "sector": {
                    "type": "string",
                    "description": (
                        "Business sector. One of: oil_gas, coal, utilities, steel, cement, "
                        "chemicals, mining, agriculture, automotive, real_estate, financials, "
                        "technology, healthcare, retail, food_beverage, transport"
                    )
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "ISO 2-letter country code where the company is headquartered (e.g. 'US', 'IN', 'GB', 'DE')"
                },
                "revenue_usd_m": {
                    "type": "number",
                    "description": "Annual revenue in USD millions. Use 5000 if unknown."
                }
            },
            "required": ["company_name", "sector", "jurisdiction"]
        }
    }
}


def _exec_assess_company(args: dict) -> dict:
    try:
        from ..data.parameters import (
            country_to_climate_zone,
            normalise_sector,
            compute_composite_score,
            get_sector_transition_params,
        )
        from ..ml.risk_predictor import RiskPredictor

        company_name   = args["company_name"]
        sector         = normalise_sector(args.get("sector", "default"))
        jurisdiction   = args.get("jurisdiction", "US")
        revenue_usd_m  = float(args.get("revenue_usd_m", 5000.0))
        climate_zone   = country_to_climate_zone(jurisdiction)
        trans_p        = get_sector_transition_params(sector)

        _cp_2030 = {"nze": 130.0, "delayed": 45.0, "cp": 18.0}
        scores = {}
        for scen in ("nze", "delayed", "cp"):
            scores[scen] = compute_composite_score(
                sector=sector, climate_zone=climate_zone,
                scenario=scen, carbon_price_usd=_cp_2030[scen], year=2030,
            )

        composite_now = scores["delayed"].get("composite_score", 50.0)

        # ND-GAIN + CarbonLens
        hf = {}
        try:
            from ..connectors.hf_carbontlens import HFEnricher
            hf = HFEnricher.enrich(country=None, jurisdiction=jurisdiction)
        except Exception:
            pass

        # Monte Carlo VaR
        var_result = {}
        try:
            from ..engine.climate_var import compute_climate_var
            vr = compute_climate_var(
                physical_score=scores["delayed"].get("physical_score", composite_now * 0.7),
                transition_score=scores["delayed"].get("transition_score", composite_now * 0.3),
                portfolio_value_usd_m=revenue_usd_m * 2.5,
                sector=sector,
                ndgain_vulnerability=hf.get("vulnerability_score"),
                ndgain_readiness=hf.get("readiness_score"),
            )
            var_result = {
                "var_95_usd_m": vr.var_95_usd_m,
                "cvar_95_usd_m": vr.cvar_95_usd_m,
                "expected_loss_usd_m": vr.expected_loss_usd_m,
                "haircut_pct": vr.haircut_pct,
                "risk_tier": vr.risk_tier,
            }
        except Exception:
            pass

        # ML trajectory
        traj = {}
        try:
            t = RiskPredictor().predict(
                base_cri_score=composite_now, sector=sector, jurisdiction=jurisdiction,
                revenue_usd_m=revenue_usd_m,
                scope1_mt_co2e=trans_p["carbon_intensity_t_per_m_rev"] * revenue_usd_m * 0.6 / 1e6,
                eal_p50_usd_m=revenue_usd_m * 0.025, wacc_adjusted=0.09,
            )
            traj = {
                "years": list(range(2025, 2051)),
                "nze": t.nze_scores,
                "delayed": t.delayed_scores,
                "cp": t.cp_scores,
                "peak_risk_year": t.peak_risk_year,
                "inflection_year": t.inflection_year,
            }
        except Exception:
            pass

        return {
            "company": company_name,
            "sector": sector,
            "jurisdiction": jurisdiction,
            "climate_zone": climate_zone,
            "composite_score": round(composite_now, 1),
            "scenario_scores": {
                s: {
                    "composite": round(v.get("composite_score", 0), 1),
                    "physical":  round(v.get("physical_score", 0), 1),
                    "transition": round(v.get("transition_score", 0), 1),
                } for s, v in scores.items()
            },
            "climate_var": var_result,
            "ndgain": {
                "gain_score": hf.get("gain_score"),
                "vulnerability": hf.get("vulnerability_score"),
                "readiness": hf.get("readiness_score"),
                "source": hf.get("source"),
            },
            "trajectory": traj,
            "sector_params": {
                "carbon_intensity": trans_p.get("carbon_intensity_t_per_m_rev"),
                "policy_risk": trans_p.get("policy_risk_score"),
                "stranded_asset_risk": trans_p.get("stranded_asset_risk"),
            },
        }
    except Exception as exc:
        logger.exception("assess_company failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. analyze_disclosure
# ─────────────────────────────────────────────────────────────────────────────

_DISCLOSURE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "analyze_disclosure",
        "description": (
            "Scan climate disclosure text (sustainability report, TCFD statement, "
            "annual report excerpt) and extract: commitment score, specificity, "
            "credibility gap flag, implied temperature rise, and key evidence quotes. "
            "Uses ClimateBERT NLP + keyword classifier fallback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The climate disclosure text to analyse (up to ~5000 words)"
                },
                "company_name": {
                    "type": "string",
                    "description": "Company name for context (optional)"
                }
            },
            "required": ["text"]
        }
    }
}


def _exec_analyze_disclosure(args: dict) -> dict:
    try:
        from ..connectors.hf_nlp_analyzer import ClimateNLPAnalyzer
        text = args["text"]
        company = args.get("company_name", "")
        result = ClimateNLPAnalyzer().analyze_text(text)
        return {
            "sentiment": result.get("sentiment"),
            "risk_score": result.get("risk_score"),
            "commitment_score": result.get("commitment_score"),
            "key_themes": result.get("key_themes", []),
            "credibility_gap": result.get("credibility_gap", False),
            "model_used": result.get("model_used", "keyword"),
            "company": company,
        }
    except Exception as exc:
        logger.exception("analyze_disclosure failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. get_location_hazard
# ─────────────────────────────────────────────────────────────────────────────

_HAZARD_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_location_hazard",
        "description": (
            "Get a live climate hazard report for a specific geographic location "
            "(latitude/longitude). Returns current hazard conditions from the "
            "Climate Risk Monitoring HuggingFace Space: temperature anomaly, "
            "precipitation, wind speed, flood risk, drought index."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude (e.g. 51.5 for London, 28.6 for New Delhi)"
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude (e.g. -0.1 for London, 77.2 for New Delhi)"
                },
                "location_name": {
                    "type": "string",
                    "description": "Human-readable location name for context"
                }
            },
            "required": ["lat", "lon"]
        }
    }
}


def _exec_get_location_hazard(args: dict) -> dict:
    try:
        from ..connectors.hf_climate_monitor import ClimateMonitorClient
        lat = float(args["lat"])
        lon = float(args["lon"])
        result = ClimateMonitorClient().get_risk_report(lat, lon)
        return {
            "location": args.get("location_name", f"{lat},{lon}"),
            "lat": lat,
            "lon": lon,
            **result,
        }
    except Exception as exc:
        logger.exception("get_location_hazard failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. get_scenario_explanation
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_scenario_explanation",
        "description": (
            "Return the carbon price paths, temperature trajectories, and commodity "
            "demand curves for the three NGFS Phase 4 scenarios: NZE 2050, "
            "Delayed Transition, and Current Policies. Use when the user asks "
            "about scenarios, warming pathways, or carbon price assumptions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "enum": ["nze", "delayed", "cp", "all"],
                    "description": "Which scenario to explain. Use 'all' for a comparison."
                }
            },
            "required": ["scenario"]
        }
    }
}


def _exec_get_scenario_explanation(args: dict) -> dict:
    scenario = args.get("scenario", "all")
    data = {
        "nze": {
            "name": "Net Zero by 2050 (NZE)",
            "ngfs_family": "NGFS Phase 4 — Net Zero 2050",
            "ssp_equivalent": "SSP1-2.6",
            "temperature_2100_c": "1.4–1.6°C",
            "carbon_price_2030_usd": 130,
            "carbon_price_2050_usd": 250,
            "description": (
                "Front-loaded, rapid decarbonisation compatible with Paris 1.5°C. "
                "Thermal coal demand collapses 90% by 2050. Crude oil -65%. "
                "Copper demand +120% (electrification). Requires immediate policy action."
            ),
            "key_risk": "Transition risk dominant — stranded fossil assets, high carbon cost.",
        },
        "delayed": {
            "name": "Delayed Transition",
            "ngfs_family": "NGFS Phase 4 — Delayed Transition",
            "ssp_equivalent": "SSP2-4.5",
            "temperature_2100_c": "1.8–2.5°C",
            "carbon_price_2030_usd": 45,
            "carbon_price_2050_usd": 240,
            "description": (
                "Policy inaction until ~2030, then abrupt emergency repricing. "
                "Carbon price jumps from $45 to $155 between 2030 and 2036. "
                "Highest transition shock risk due to abrupt policy change. "
                "More physical risk than NZE due to delayed action."
            ),
            "key_risk": "Both physical and transition risks elevated — worst of both worlds.",
        },
        "cp": {
            "name": "Current Policies",
            "ngfs_family": "NGFS Phase 4 — Current Policies",
            "ssp_equivalent": "SSP3-7.0",
            "temperature_2100_c": "2.5–3.8°C",
            "carbon_price_2030_usd": 18,
            "carbon_price_2050_usd": 75,
            "description": (
                "No additional climate policy. Physical risks dominate and accelerate. "
                "Fossil fuel demand remains elevated. Heat stress, floods, cyclones "
                "intensify significantly by 2050. Financial system stress from physical damage."
            ),
            "key_risk": "Physical risk dominant — asset damage, supply chain disruption, stranded coastal assets.",
        }
    }
    if scenario == "all":
        return {"scenarios": data}
    return data.get(scenario, {"error": f"Unknown scenario: {scenario}"})


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_ticker_data
# ─────────────────────────────────────────────────────────────────────────────

_TICKER_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_ticker_data",
        "description": (
            "Look up CarbonLens climate risk data for a stock ticker (S&P 500 companies). "
            "Returns credit score, risk tier, ESG level, carbon intensity flag, and "
            "ND-GAIN country scores. Works for ~421 S&P 500 tickers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. 'AAPL', 'XOM', 'JPM', 'TSLA')"
                }
            },
            "required": ["ticker"]
        }
    }
}


def _exec_get_ticker_data(args: dict) -> dict:
    try:
        from ..connectors.hf_carbontlens import HFEnricher
        ticker = args["ticker"].upper().strip()
        result = HFEnricher.by_ticker(ticker)
        if result:
            return {"ticker": ticker, "found": True, **result}
        return {
            "ticker": ticker,
            "found": False,
            "note": "Ticker not in CarbonLens database (covers ~421 S&P 500 companies)"
        }
    except Exception as exc:
        logger.exception("get_ticker_data failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. compare_companies
# ─────────────────────────────────────────────────────────────────────────────

_COMPARE_DEFINITION = {
    "type": "function",
    "function": {
        "name": "compare_companies",
        "description": (
            "Compare climate risk scores for two companies side by side. "
            "Returns composite scores, physical/transition breakdown, VaR, and "
            "a relative ranking for each NGFS scenario. Use when the user asks "
            "'which is riskier' or wants a peer comparison."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_a": {
                    "type": "object",
                    "description": "First company",
                    "properties": {
                        "name": {"type": "string"},
                        "sector": {"type": "string"},
                        "jurisdiction": {"type": "string"},
                        "revenue_usd_m": {"type": "number"}
                    },
                    "required": ["name", "sector", "jurisdiction"]
                },
                "company_b": {
                    "type": "object",
                    "description": "Second company",
                    "properties": {
                        "name": {"type": "string"},
                        "sector": {"type": "string"},
                        "jurisdiction": {"type": "string"},
                        "revenue_usd_m": {"type": "number"}
                    },
                    "required": ["name", "sector", "jurisdiction"]
                }
            },
            "required": ["company_a", "company_b"]
        }
    }
}


def _exec_compare_companies(args: dict) -> dict:
    results = {}
    for key in ("company_a", "company_b"):
        co = args[key]
        res = _exec_assess_company({
            "company_name": co["name"],
            "sector": co["sector"],
            "jurisdiction": co["jurisdiction"],
            "revenue_usd_m": co.get("revenue_usd_m", 5000.0),
        })
        results[key] = res

    a = results["company_a"]
    b = results["company_b"]

    if "error" in a or "error" in b:
        return {"error": "One or both assessments failed", "details": results}

    a_score = a.get("composite_score", 50)
    b_score = b.get("composite_score", 50)
    riskier = args["company_a"]["name"] if a_score > b_score else args["company_b"]["name"]
    delta   = abs(a_score - b_score)

    return {
        "company_a": {
            "name": args["company_a"]["name"],
            "composite_score": a_score,
            "scenario_scores": a.get("scenario_scores", {}),
            "var_95_usd_m": (a.get("climate_var") or {}).get("var_95_usd_m"),
            "risk_tier": (a.get("climate_var") or {}).get("risk_tier"),
        },
        "company_b": {
            "name": args["company_b"]["name"],
            "composite_score": b_score,
            "scenario_scores": b.get("scenario_scores", {}),
            "var_95_usd_m": (b.get("climate_var") or {}).get("var_95_usd_m"),
            "risk_tier": (b.get("climate_var") or {}).get("risk_tier"),
        },
        "verdict": {
            "higher_risk": riskier,
            "score_delta": round(delta, 1),
            "interpretation": (
                f"{riskier} has a {round(delta,1)}-point higher composite climate risk score. "
                f"{'This is a material difference.' if delta > 10 else 'The scores are relatively close.'}"
            ),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. get_country_risk
# ─────────────────────────────────────────────────────────────────────────────

_COUNTRY_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_country_risk",
        "description": (
            "Get ND-GAIN country climate risk scores for a country. Returns gain_score "
            "(0–100), vulnerability (0–1), and readiness (0–1). Higher vulnerability = "
            "more exposed to climate impacts. Lower readiness = less able to adapt. "
            "Covers 59 countries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "jurisdiction": {
                    "type": "string",
                    "description": "ISO 2-letter country code (e.g. 'IN', 'CN', 'NG', 'BD')"
                }
            },
            "required": ["jurisdiction"]
        }
    }
}


def _exec_get_country_risk(args: dict) -> dict:
    try:
        from ..connectors.hf_carbontlens import HFEnricher
        iso2 = args["jurisdiction"].upper()
        result = HFEnricher.by_jurisdiction(iso2)
        if result:
            return {"jurisdiction": iso2, "found": True, **result}
        return {
            "jurisdiction": iso2,
            "found": False,
            "note": "Country not in ND-GAIN extended dataset (59 countries covered)"
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 8. assess_physical_asset
# ─────────────────────────────────────────────────────────────────────────────

_ASSET_DEFINITION = {
    "type": "function",
    "function": {
        "name": "assess_physical_asset",
        "description": (
            "Evidence-based physical climate risk for a real asset at GPS coordinates. "
            "Analyses the 5 km x 5 km area around the site using: a 10 m terrain model "
            "(AWS terrain tiles) with D8 flow-routing, slope and terrain position; real "
            "Sentinel-2 satellite pixels (NDVI, water index, optical water history); Sentinel-1 "
            "radar flood history across ~10 dates in 2 years (sees through monsoon cloud); and ESA "
            "WorldCover 10 m land cover (built-up, flammable vegetation, wetlands); and a "
            "global tropical-cyclone wind model (NOAA IBTrACS 1980–2025, validated against "
            "station winds) with 1-in-50/100-year winds and the worst historical storm. Returns "
            "findings for flood, debris flow, landslide, wildfire, urban heat and tropical cyclone — each with "
            "severity, the measured evidence and its source, and site-specific remediation "
            "actions. Takes ~20-40 s. Use when the user gives a location, address, lat/lon, "
            "building, factory, warehouse, port, mine or farm, or asks about site-level risk. "
            "Cite the evidence values in your answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude of the asset in decimal degrees (e.g. -33.8688 for Sydney)"
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude of the asset in decimal degrees (e.g. 151.2093 for Sydney)"
                },
                "asset_name": {
                    "type": "string",
                    "description": "Name or description of the asset (e.g. 'Sydney Distribution Centre', 'Collie Power Station')"
                },
                "asset_type": {
                    "type": "string",
                    "description": (
                        "Type of asset. One of: industrial, warehouse, logistics, port, "
                        "office, retail, agriculture, data_centre, hospital, residential. "
                        "Determines asset-specific remediation addons."
                    )
                }
            },
            "required": ["lat", "lon"]
        }
    }
}


def _exec_assess_physical_asset(args: dict) -> dict:
    try:
        from ..analysis.asset_intelligence import assess_asset
        r = assess_asset(
            lat=float(args["lat"]),
            lon=float(args["lon"]),
            asset_name=args.get("asset_name", "Asset"),
            asset_type=args.get("asset_type", "industrial"),
            include_visuals=False,   # never push imagery/grids into the LLM context
        )
        # Compact view for the LLM: findings + headline, not raw sub-results
        return {
            "asset_name": r["asset_name"], "lat": r["lat"], "lon": r["lon"],
            "risk_signal": r["risk_signal"], "headline": r["headline"],
            "findings": [
                {"hazard": f["hazard"], "severity": f["severity"],
                 "evidence": [f"{e['metric']}: {e['value']}{(' ' + e['unit']) if e['unit'] else ''} ({e['source']})"
                              for e in f["evidence"]],
                 "actions": f["actions"][:4],
                 **({"caveat": f["caveat"]} if f.get("caveat") else {})}
                for f in r["findings"] if f["severity"] != "NEGLIGIBLE"
            ],
            "satellite_scene_date": r["satellite"].get("scene_date"),
            "optical_water_history_window": r["satellite"].get("temporal_date_range"),
            "radar_water_history_window": (r.get("radar") or {}).get("date_range"),
            "sources": r["sources"],
            "caveats": r["caveats"][:2],
        }
    except Exception as exc:
        logger.exception("assess_physical_asset failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 9. get_company_assets
# ─────────────────────────────────────────────────────────────────────────────

_COMPANY_ASSETS_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_company_assets",
        "description": (
            "Find where a company's physical assets are (power plants, cement, steel, refineries, "
            "chemical plants, mines — Climate TRACE global + EPA US registers) and how exposed they "
            "are: precomputed tropical-cyclone wind (validated model) and low-lying flood screening "
            "per asset, plus portfolio shares. Instant. Use when asked where a company operates, "
            "which of its sites are most at risk, or for portfolio physical exposure. Follow up "
            "with assess_physical_asset on specific sites for the full satellite analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Company name, e.g. 'Holcim', 'NTPC', 'ExxonMobil'"},
                "include_subsidiaries": {"type": "boolean",
                                         "description": "Also include owners whose names start with the same words (default true)"}
            },
            "required": ["company_name"]
        }
    }
}


def _exec_get_company_assets(args: dict) -> dict:
    try:
        from ..data.asset_register import company_exposure, norm, search
        res = search(args["company_name"], limit=20)
        if not res:
            return {"found": False, "note": "No owner in the asset register (heavy industry only)."}
        q = norm(args["company_name"])
        keys = [r["key"] for r in res if r["match"] in ("exact", "prefix")] if args.get("include_subsidiaries", True) \
            else [res[0]["key"]]
        keys = keys or [res[0]["key"]]
        e = company_exposure(keys)
        top = [{k: a[k] for k in ("name", "country", "sector", "lat", "lon", "screen_level", "elev_m")}
               | {"cyclone_1in100": a["cyclone"]["rp100_category"], "worst_storm": a["cyclone"]["worst"]}
               for a in e["assets"][:10]]
        return {"found": True, "entities": e["companies"], "n_assets": e["n_assets"],
                "summary": {k: v for k, v in e["summary"].items() if k != "sectors"},
                "most_exposed_assets": top, "method": e["method"], "caveats": e["caveats"]}
    except Exception as exc:
        logger.exception("get_company_assets failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 10. run_risk_analyst — physical → policy → transition → financial distribution
# ─────────────────────────────────────────────────────────────────────────────

_ANALYST_DEFINITION = {
    "type": "function",
    "function": {
        "name": "run_risk_analyst",
        "description": (
            "Full four-stage analysis of ONE asset (by coordinates) or ONE company (by name): "
            "(1) physical hazards — cyclone, river and coastal flood depths with local flood defences; "
            "(2) carbon-pricing policies in force and in the pipeline for the asset's state/country and sector; "
            "(3) NGFS Phase 5 transition projections (carbon price, sector output) for 7 scenarios; "
            "(4) Monte Carlo financial loss distribution to 2050 (mean, median, 95th/99th percentile, "
            "expected shortfall, % of asset value) plus findings and actions. Use for 'what is the "
            "financial climate risk of X', 'how do current policies affect this plant', or loss distributions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Asset latitude (use with lon for a single asset)"},
                "lon": {"type": "number", "description": "Asset longitude"},
                "company_name": {"type": "string", "description": "Company name for a portfolio analysis (instead of lat/lon)"},
                "value_usd": {"type": "number", "description": "Asset value in USD if known"},
            },
        },
    },
}


def _compact_analysis(r: dict) -> dict:
    fin = r["financial"]
    scen = {v["label"]: {"total_mean": v["total_npv"]["mean"], "total_p95": v["total_npv"]["p95"],
                         "carbon_cost_pv": v["transition_npv"]["mean"],
                         "pct_of_value_mean": (v["total_npv"].get("pct_of_value") or {}).get("mean")}
            for v in fin["scenarios"].values()}
    return {"asset": {k: r["asset"][k] for k in ("name", "iso3", "admin1", "sector", "type", "emissions_t")}
                     | {"value_basis": r["asset"]["value"].get("basis"), "value_usd": r["asset"]["value"].get("mode")},
            "headline": r["headline"], "narrative": r["narrative"],
            "physical_severity": {"cyclone": r["physical"]["cyclone"]["severity"],
                                  "river_flood": r["physical"]["river_flood"].get("severity"),
                                  "coastal_flood": r["physical"]["coastal_flood"].get("severity")},
            "policies_in_force": [{k: p[k] for k in ("name", "type", "price_usd_t", "covers_asset_sector")}
                                  for p in r["policy"]["in_force"]],
            "policies_pipeline": [p["name"] for p in r["policy"]["pipeline"]],
            "policy_notes": r["policy"]["notes"],
            "scenarios_pv_usd_2026_2050": scen, "actions": r["actions"], "caveats": r["caveats"]}


def _exec_run_risk_analyst(args: dict) -> dict:
    try:
        from ..finance.analyst import analyse_asset, analyse_company
        if args.get("lat") is not None and args.get("lon") is not None:
            return _compact_analysis(analyse_asset(float(args["lat"]), float(args["lon"]), value_usd=args.get("value_usd"), n=2000))
        if args.get("company_name"):
            from ..data.asset_register import search
            res = search(args["company_name"], limit=10)
            if not res:
                return {"found": False, "note": "Company not in the asset register (heavy industry only)."}
            keys = [r["key"] for r in res if r["match"] == "group"] or [res[0]["key"]]
            c = analyse_company(keys, top_n=15, n=1500)
            return {"companies": c.get("companies"), "n_assets": c.get("n_assets"),
                    "portfolio_value_usd": c.get("portfolio_value_mode"),
                    "scenarios_pv_usd_2026_2050": {v["label"]: {k: v["total_npv"][k] for k in ("mean", "p50", "p95", "es95")}
                                                   for v in c.get("scenarios", {}).values()},
                    "top_assets": c.get("assets", [])[:8], "method": c.get("method")}
        return {"error": "Give lat/lon for an asset or company_name for a portfolio."}
    except Exception as exc:
        logger.exception("run_risk_analyst failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict] = {
    "assess_company":          {"definition": _ASSESS_DEFINITION,    "executor": _exec_assess_company},
    "analyze_disclosure":      {"definition": _DISCLOSURE_DEFINITION, "executor": _exec_analyze_disclosure},
    "get_location_hazard":     {"definition": _HAZARD_DEFINITION,     "executor": _exec_get_location_hazard},
    "get_scenario_explanation":{"definition": _SCENARIO_DEFINITION,  "executor": _exec_get_scenario_explanation},
    "get_ticker_data":         {"definition": _TICKER_DEFINITION,     "executor": _exec_get_ticker_data},
    "compare_companies":       {"definition": _COMPARE_DEFINITION,    "executor": _exec_compare_companies},
    "get_country_risk":        {"definition": _COUNTRY_DEFINITION,    "executor": _exec_get_country_risk},
    "assess_physical_asset":   {"definition": _ASSET_DEFINITION,     "executor": _exec_assess_physical_asset},
    "get_company_assets":      {"definition": _COMPANY_ASSETS_DEFINITION, "executor": _exec_get_company_assets},
    "run_risk_analyst":        {"definition": _ANALYST_DEFINITION,   "executor": _exec_run_risk_analyst},
}


def get_all_definitions() -> list[dict]:
    """Return all tool definitions for injection into the Groq tools= array."""
    return [v["definition"] for v in TOOL_REGISTRY.values()]


def execute_tool(name: str, args: dict) -> dict:
    """Execute a named tool and return its result dict."""
    entry = TOOL_REGISTRY.get(name)
    if not entry:
        return {"error": f"Unknown tool: {name}"}
    try:
        return entry["executor"](args)
    except Exception as exc:
        logger.exception("Tool execution failed: %s", name)
        return {"error": str(exc)}
