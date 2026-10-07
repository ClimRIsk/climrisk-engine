"""Climate Risk Intelligence (CRI) engine.

Translating physical and transition climate risks into quantifiable
financial outcomes at the company and portfolio level.
"""

__version__ = "0.4.1"


def _trust_store() -> None:
    """Make every HTTPS fetch (urllib, requests) verify against the certifi CA bundle.

    A python.org Python on macOS ships without root certificates until the user runs
    'Install Certificates.command'. Without them every urllib call (terrain tiles, Sentinel
    search, GDACS, ONI) fails with CERTIFICATE_VERIFY_FAILED, while rasterio/GDAL reads (which
    bring their own bundle) still work — so the 3D model came up flat and half the layers went
    grey. Pointing Python at certifi's bundle removes the dependency on that manual step.
    Never overrides a bundle the operator has set, and never raises.
    """
    import os
    try:
        import certifi
        path = certifi.where()
    except Exception:                                              # noqa: BLE001
        return
    if not os.path.exists(path):
        return
    os.environ.setdefault("SSL_CERT_FILE", path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
    try:
        import ssl
        cafile = os.environ.get("SSL_CERT_FILE", path)
        ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=cafile)   # type: ignore[assignment]
    except Exception:                                              # noqa: BLE001
        pass


_trust_store()
