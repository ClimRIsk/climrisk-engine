"""For hazards the engine has no verified solver for: research it, say which physics and public datasets apply, and show exactly what is missing. Never invent a number."""
from __future__ import annotations

# hazard → (label, aliases regex, physics lines, datasets [(name, url, what)], what a model needs, nearest capability inside the engine)
CATALOG = {
    "surge": ("Storm surge", r"storm[\s\-]*surge|coastal surge|sea[\s\-]*level (?:rise|extreme)|coastal flood\w*|tide\w*",
              ["Wind stress piles water against the coast, and low pressure lifts the sea surface by about 1 cm per hPa (the inverse-barometer effect).", "The depth-averaged shallow-water equations with a wind-stress term, **τ = ρ_air C_d |U| U**, on a coastal bathymetry grid, are the standard physics (the engine's flood solver already solves the same equations)."],
              [("GEBCO bathymetry", "https://www.gebco.net", "sea-floor depth for the surge grid"), ("UHSLC / PSMSL tide gauges", "https://uhslc.soest.hawaii.edu", "observed sea level for validation"), ("IBTrACS", "https://www.ncei.noaa.gov/products/international-best-track-archive", "storm tracks and pressure (bundled)")],
              "A coastal bathymetry grid, a wind-stress term in the flood solver, a tide model, and tide-gauge records to validate against.", "The cyclone study gives winds and pressure along each track; the flood solver gives the shallow-water engine."),
    "tsunami": ("Tsunami", r"tsunami", ["Long ocean waves travel at **c = √(g h)**; the linear shallow-water equations propagate them from an earthquake-source displacement (Okada 1985) and the wave amplifies on shoaling (Green's law, amplitude ∝ h^(−1/4))."],
                [("GEBCO bathymetry", "https://www.gebco.net", "ocean depth"), ("NOAA NCEI tsunami database", "https://www.ngdc.noaa.gov/hazel/view/hazards/tsunami/event-data", "past events and run-ups for validation"), ("USGS ComCat", "https://earthquake.usgs.gov", "source earthquakes")],
                "An ocean-basin bathymetry grid, an earthquake-source deformation model, and the NOAA run-up database to validate against.", "The earthquake study gives source rates and magnitudes; the flood solver gives the wave propagation numerics."),
    "volcano": ("Volcanic ash and eruption", r"volcan\w*|eruption|ash fall", ["Eruption frequency follows a Poisson process per volcano; ash dispersion is an advection–diffusion problem with particle settling (Suzuki 1983 source, terminal fall velocity ~ d²)."],
                [("Smithsonian Global Volcanism Program", "https://volcano.si.edu", "eruption history for rates and validation"), ("ERA5 winds", "https://cds.climate.copernicus.eu", "upper-air wind for ash transport")],
                "A volcano inventory near the site, an eruption-size distribution, and upper-air winds with an advection–diffusion transport model.", "The air-quality dispersion code is the nearest numerical relative."),
    "hail": ("Hail", r"\bhail\w*", ["Hail forms where strong updrafts hold ice in supercooled water; proxies use convective available potential energy (CAPE) and wind shear."],
             [("ERA5 CAPE and shear", "https://cds.climate.copernicus.eu", "environmental proxy"), ("NOAA Storm Events database / ESWD", "https://www.ncei.noaa.gov/stormevents/", "reports for validation (US/Europe only)")],
             "CAPE and shear from reanalysis, a hail-proxy index calibrated against storm reports, which exist mainly for the US and Europe.", "The extreme-value framework could test an index once one is calibrated."),
    "tornado": ("Tornado", r"tornado\w*", ["Tornado environments combine large CAPE with strong low-level shear; an index such as the Significant Tornado Parameter is used as a proxy."],
                [("NOAA SPC tornado database", "https://www.spc.noaa.gov/gis/svrgis/", "US tornado tracks")], "A reanalysis proxy and a tornado report archive; only the US has a dense one.", "Extreme-value hindcast framework."),
    "subsidence": ("Land subsidence", r"subsidence|sinking|land sinking", ["Compaction follows Terzaghi consolidation, **ΔH = H · Cc/(1+e0) · log10(σ′f/σ′0)**, driven by groundwater withdrawal or load."],
                   [("Sentinel-1 InSAR (EGMS / COMET LiCSAR)", "https://comet.nerc.ac.uk/COMET-LiCSAR-portal/", "measured ground motion for validation")], "InSAR velocity maps for validation and groundwater-level records or a soil column to drive the model.", "None: needs a new compaction solver."),
    "avalanche": ("Snow avalanche", r"avalanche\w*", ["Slab release is an infinite-slope shear balance on a weak snow layer, similar to the engine's landslide model with snowpack properties."],
                  [("ERA5-Land snow depth", "https://cds.climate.copernicus.eu", "snowpack")], "A snow-layer stability model and avalanche records.", "Landslide infinite-slope solver."),
    "dust": ("Dust storm", r"dust[\s\-]*storm\w*|sand[\s\-]*storm\w*", ["Dust emission starts above a threshold friction velocity (Shao & Lu 2000) and scales with its cube."],
             [("CAMS reanalysis dust", "https://ads.atmosphere.copernicus.eu", "dust aerosol optical depth")], "Soil texture and moisture maps and the CAMS dust record to calibrate emission.", "Air-quality dispersion code."),
    "lightning": ("Lightning", r"lightning", ["Flash rate scales with cloud-top height and convective updraft strength."], [("ERA5 CAPE", "https://cds.climate.copernicus.eu", "proxy"), ("WWLLN / GLM", "https://www.nasa.gov", "flash counts")], "A flash-rate proxy calibrated on a lightning network.", "None."),
    "erosion": ("Coastal erosion", r"coastal erosion|shoreline|beach erosion", ["Shoreline change follows longshore sediment transport (CERC formula **Q = K Hb^{5/2} sin 2αb**) and cross-shore profile response."],
                [("Landsat/Sentinel-2 shoreline (CoastSat)", "https://github.com/kvos/CoastSat", "observed shoreline history")], "Wave climate and a shoreline-change record from satellite images.", "None."),
}


def match(text: str):
    import re
    best = None
    for k, v in CATALOG.items():
        m = re.search(v[1], text, re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), k)
    return best[1] if best else None


def learn_hazard(key: str, name: str = "", lat: float = None, lon: float = None, progress=None, research: bool = True) -> dict:
    """Return a standard result dict that says: what physics applies, which datasets exist, what is missing. Tries to read up on the topic first."""
    from .generic_report import Notebook
    label, _, physics, datasets, needs, nearest = CATALOG[key]
    nb = Notebook(progress)
    nb("1", "Problem", f"The engine was asked to model {label.lower()}" + (f" for {name}" if name else "") + ". It has no verified solver for this hazard yet.")
    nb("2", "Self-assessment", "Writing a new solver is only useful if it can be verified and validated. The engine checked what it would need and what it has.")
    basis_note = ""
    if research:
        try:
            from ..resident import knowledge as K
            ps = K.search(label, 2)
            if (not ps or ps[0]["coverage"] < 0.5):
                K.learn_topic(label)
                ps = K.search(label, 2)
            if ps and ps[0]["coverage"] >= 0.5:
                basis_note = f'{ps[0]["title"]}: ' + " ".join(ps[0]["text"].split()[:40]) + "…"
                nb("3", "Research", f'Read up on {label.lower()} from {ps[0]["title"]} ({ps[0]["url"]}).')
        except Exception:                                                   # noqa: BLE001
            pass
    nb("4", "Gap analysis", f"Missing: {needs}")
    blocks = [("h2", "question", "The question"), ("p", f"Can the engine build and validate a {label.lower()} model" + (f" for {name}" if name else "") + "?"),
              ("h2", "physics", "The physics that applies")] + [("p", t) for t in physics]
    if basis_note:
        blocks.append(("p", "From the engine's reading: " + basis_note))
    blocks += [("h2", "data", "Public data that exist"), ("table", ["Dataset", "What it provides", "Link"], [[a, c, b] for a, b, c in datasets], "All free to use."),
               ("h2", "result", "What stops the engine today"), ("callout", "No model was run, so there are no numbers. The engine does not guess a hazard it cannot verify.", "warn"), ("p", needs),
               ("p", f"Nearest capability already in the engine: {nearest}")]
    return {"domain": key, "label": label, "kind": "dossier", "name": name, "lat": lat, "lon": lon, "title": f"{label}: what the engine knows and what it would take to model it",
            "standfirst": f"The engine has no verified solver for {label.lower()} yet. It identified the governing physics and {len(datasets)} public datasets that could validate a model, and listed what is missing. No numbers were produced.",
            "status": "no model yet (dossier only)", "ok": True, "headline": [("none", "no model run"), (str(len(datasets)), "public datasets identified")], "blocks": blocks, "verification": [], "sensitivity": [],
            "assumptions": [], "limitations": ["This is a research dossier, not a hazard assessment."], "next_steps": [needs], "notebook": nb.rows, "numbers": {}, "elapsed_s": 0.0}
