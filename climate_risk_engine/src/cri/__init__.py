"""Climate Risk Intelligence (CRI) engine.

Translating physical and transition climate risks into quantifiable
financial outcomes at the company and portfolio level.
"""

__version__ = "0.6.1"


def _trust_store() -> None:
    """Make every HTTPS fetch (urllib, requests, http.client) verify against a working CA set.

    A python.org Python on macOS ships without root certificates until the user runs
    'Install Certificates.command'. Without them every urllib call (terrain tiles, Sentinel
    search, GDACS, ONI) fails with CERTIFICATE_VERIFY_FAILED, while rasterio/GDAL reads (which
    bring their own bundle) still work — so the 3D model came up flat and half the layers went
    grey. Order of preference:
      1. truststore  — the operating-system store (macOS Keychain), which also covers networks that
                       re-sign traffic with their own root ("self-signed certificate in chain");
      2. certifi     — a bundled Mozilla CA list.
    The default-context factory is patched on BOTH ssl and http.client, because http.client keeps its
    own reference taken at import time: a program that imported urllib first would otherwise keep the
    unpatched one. Never overrides a bundle the operator has set, and never raises.
    """
    import os
    try:
        import truststore
        truststore.inject_into_ssl()
        return
    except Exception:                                              # noqa: BLE001
        pass
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

        def _ctx(*a, **k):
            return ssl.create_default_context(cafile=cafile)

        ssl._create_default_https_context = _ctx                   # type: ignore[assignment]
        import http.client
        if hasattr(http.client, "_create_default_https_context"):
            http.client._create_default_https_context = _ctx       # type: ignore[attr-defined]
    except Exception:                                              # noqa: BLE001
        pass


_trust_store()
