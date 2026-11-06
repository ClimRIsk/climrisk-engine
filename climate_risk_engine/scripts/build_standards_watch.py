"""Build src/cri/data/standards_watch.json: the status of the standards and rules the engine's methods follow.

Every status below was read on 6 Nov 2026 from the publisher's own page (URLs in each item). Nothing here is quoted from the paywalled
text of a standard: ISO 14067 is tracked by edition, stage and dates only. Anything not read from a primary page is listed under
`not_assessed` instead of being filled in.

Run:  PYTHONPATH=src python scripts/build_standards_watch.py     (writes the JSON; the test suite fails if the JSON differs from what this writes)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "cri" / "data" / "standards_watch.json"
AS_OF = "2026-11-06"

STATUS = {
    "under_revision": "in force; a revision is under way",
    "amended_effective_later": "in force; an amendment is issued and takes effect later",
    "adopted_pending_scrutiny": "a replacement is adopted by the Commission and awaits Parliament/Council scrutiny",
    "in_application": "in application",
    "newer_edition_published": "a newer edition is published",
}
ENGINE_STATE = {"matches", "partly", "behind"}

# ── items ────────────────────────────────────────────────────────────────────

ITEMS = [
    {
        "id": "ghgp", "name": "GHG Protocol corporate standards", "body": "GHG Protocol (WRI and WBCSD) with ISO",
        "covers": "Corporate Standard (2004), Scope 3 Standard (2011), Scope 2 Guidance (2015), and the Actions and Market Instruments (AMI) workstream",
        "status": "under_revision",
        "headline": "The 2004/2011/2015 texts remain in force. GHG Protocol and ISO are merging them with ISO 14064-1 into one co-branded corporate standard: integrated public consultation planned for Q2 2027, publication planned for Q4 2028.",
        "key_dates": [
            {"date": "2025-09", "done": True, "event": "GHG Protocol and ISO announce a strategic partnership to co-develop a consolidated corporate standard"},
            {"date": "2025-10-20", "done": True, "event": "Public consultation on the Scope 2 Guidance opens"},
            {"date": "2026-01-31", "done": True, "event": "Scope 2 consultation closes"},
            {"date": "2026-07-29", "done": True, "event": "Consultation summary published (nearly 1,100 responses from 56 countries), consolidated Standard Development Plan published, AMI request-for-information feedback shared"},
            {"date": "2027-Q2", "done": False, "event": "Integrated public consultation on the consolidated corporate standard (publisher-stated plan)"},
            {"date": "2028-Q4", "done": False, "event": "Publication of the consolidated joint corporate standard (publisher-stated plan)"},
        ],
        "engine": {"state": "matches", "holds": "Scope 2 Guidance (2015): location- and market-based Scope 2",
                   "where": "carbon/factors.py, carbon/inventory.py",
                   "note": "That is the edition in force. No revised Scope 2 text exists yet: the publisher says several reporting approaches are being explored and any proposal still needs Technical Working Group development and Independent Standards Board approval."},
        "not_assessed": ["the content of any revised Scope 2 rule (no draft text is published)", "final approval dates"],
        "monitors": {"ghgp": {"edition": ["2026-07-29"], "label": "Consolidated GHG Protocol-ISO Standard Development Plan of 2026-07-29", "date": "2026-07-29"}},
        "sources": [
            {"title": "GHG Protocol: Corporate Suite update process", "url": "https://ghgprotocol.org/ghg-protocol-corporate-suite-standards-and-guidance-update-process"},
            {"title": "GHG Protocol: Key Standard Development Updates (29 Jul 2026)", "url": "https://ghgprotocol.org/blog/ghg-protocol-announces-key-standard-development-updates"},
            {"title": "GHG Protocol: FAQ resource on the 29 Jul 2026 updates", "url": "https://ghgprotocol.org/blog/ghg-protocol-announces-key-standard-development-updates-faq-resource"},
            {"title": "GHG Protocol: Scope 2 Guidance", "url": "https://ghgprotocol.org/scope-2-guidance"},
        ],
    },
    {
        "id": "iso14067", "name": "ISO 14067 carbon footprint of products", "body": "ISO/TC 207/SC 7 (and GHG Protocol-ISO Joint Working Group for product-level accounting)",
        "covers": "ISO 14067:2018, Edition 1, published August 2018",
        "status": "under_revision",
        "headline": "ISO 14067:2018 is current (reviewed and confirmed in 2024) and ISO lists it as 'International Standard to be revised' (stage 90.92). GHG Protocol and ISO named a Joint Working Group in February 2026 to build a harmonised product-level standard. No publication date for a revision is stated on the pages read.",
        "key_dates": [
            {"date": "2018-08", "done": True, "event": "ISO 14067:2018 (Edition 1) published"},
            {"date": "2024", "done": True, "event": "Reviewed and confirmed by ISO"},
            {"date": "2026-02-09", "done": True, "event": "GHG Protocol announces its nominees to the Joint Working Group with ISO for an updated, harmonised product-level standard"},
        ],
        "engine": {"state": "matches", "holds": "ISO 14067:2018 method structure (data-quality criteria, allocation hierarchy, study-report content)", "where": "lca/cfp.py",
                   "note": "The standard is paywalled, so only its edition, stage and dates are tracked here; no clause text is stored or reproduced."},
        "not_assessed": ["the stage and expected date of the revision project (the project page could not be read)", "what the revision changes"],
        "monitors": {"iso14067": {"edition": ["2018", "90.92", "2024"], "label": "ISO 14067:2018, stage 90.92 (to be revised), last confirmed 2024", "date": "2018-08-01"}},
        "sources": [
            {"title": "ISO: ISO 14067:2018", "url": "https://www.iso.org/standard/71206.html"},
            {"title": "GHG Protocol: Product Standard", "url": "https://ghgprotocol.org/product-standard"},
        ],
    },
    {
        "id": "ifrs_s2", "name": "IFRS S2 Climate-related Disclosures (ISSB)", "body": "IFRS Foundation, International Sustainability Standards Board",
        "covers": "IFRS S2 (June 2023) and the December 2025 Amendments to Greenhouse Gas Emissions Disclosures",
        "status": "amended_effective_later",
        "headline": "The ISSB issued amendments to the GHG emissions disclosure requirements of IFRS S2 in December 2025. They apply to annual periods beginning on or after 1 January 2027; early application is permitted.",
        "key_dates": [
            {"date": "2023-06", "done": True, "event": "IFRS S2 issued"},
            {"date": "2025-12", "done": True, "event": "Amendments to Greenhouse Gas Emissions Disclosures issued (with consequential amendments to the SASB Standards)"},
            {"date": "2027-01-01", "done": False, "event": "Amendments effective for annual reporting periods beginning on or after this date; early application permitted"},
        ],
        "engine": {"state": "partly", "holds": "IFRS S2 as issued in June 2023", "where": "outcomes/disclosure.py",
                   "note": "The December 2025 amendments are not reflected. I have not read what they change, so the size of the gap is not assessed."},
        "not_assessed": ["the content of the amendments", "which jurisdictions have adopted IFRS S2 and when"],
        "monitors": {"ifrs_s2": {"edition": ["2025-12", "2027-01-01"], "label": "IFRS S2 GHG amendments issued Dec 2025, effective 1 Jan 2027", "date": "2025-12-01"}},
        "sources": [{"title": "IFRS Foundation: Amendments to Greenhouse Gas Emissions Disclosures (Amendments to IFRS S2)", "url": "https://www.ifrs.org/projects/completed-projects/2025/amendments-to-disclosure-of-greenhouse-gas-emissions-s2/"}],
    },
    {
        "id": "esrs", "name": "European Sustainability Reporting Standards (CSRD)", "body": "European Commission, with technical advice from EFRAG",
        "covers": "ESRS Set 1 (Delegated Regulation (EU) 2023/2772) and the simplified ESRS adopted as a delegated act on 3 July 2026",
        "status": "adopted_pending_scrutiny",
        "headline": "The original ESRS (Official Journal, 22 Dec 2023) remain the text in force. On 3 Jul 2026 the Commission published delegated acts for simplified ESRS and a voluntary standard and sent them to Parliament and Council. If not rejected they enter into force on publication in the Official Journal; the simplified ESRS apply to financial years beginning on or after 1 Jan 2027, with early adoption for 2026 once in force. Whether they have entered into force was not checked.",
        "key_dates": [
            {"date": "2023-12-22", "done": True, "event": "ESRS Set 1 published in the Official Journal as a delegated act"},
            {"date": "2025-12-03", "done": True, "event": "EFRAG publishes its draft simplified ESRS (technical advice delivered end November 2025)"},
            {"date": "2026-05", "done": True, "event": "Commission publishes a draft delegated act for public consultation"},
            {"date": "2026-07-03", "done": True, "event": "Commission publishes the delegated acts on the revised ESRS and the voluntary standard; transmitted to Parliament and Council for scrutiny"},
            {"date": "2027-01-01", "done": False, "event": "Simplified ESRS apply to financial years beginning on or after this date (early adoption for 2026 once the act is in force)"},
        ],
        "engine": {"state": "partly", "holds": "ESRS E1 requirement-level checklist from Delegated Regulation (EU) 2023/2772", "where": "regulatory/esrs_e1_gap.py",
                   "note": "That is the version in force today. The simplified ESRS would change the requirements once in force; not reflected."},
        "not_assessed": ["whether the delegated act has entered into force (Official Journal not checked)", "the Omnibus I Directive's final scope thresholds (only secondary sources seen)", "what the simplified E1 changes"],
        "monitors": {"esrs": {"edition": ["2026-07-03"], "label": "Simplified ESRS delegated act published 3 Jul 2026, pending scrutiny", "date": "2026-07-03"}},
        "sources": [
            {"title": "EFRAG: Commission publishes delegated act on revised ESRS (3 Jul 2026)", "url": "https://www.efrag.org/en/news-and-calendar/news/european-commission-publishes-delegated-act-on-revised-esrs-and-voluntary-sustainability-reporting"},
            {"title": "EFRAG: Draft simplified ESRS", "url": "https://www.efrag.org/en/draft-simplified-esrs"},
        ],
    },
    {
        "id": "cbam", "name": "EU Carbon Border Adjustment Mechanism", "body": "European Commission (DG TAXUD)",
        "covers": "Regulation (EU) 2023/956 as simplified by Regulation (EU) 2025/2083; implementing acts on methods, default values, benchmarks and certificate prices",
        "status": "in_application",
        "headline": "The definitive regime applies from 1 Jan 2026. Quarterly certificate prices for 2026: Q1 EUR 75.36, Q2 EUR 75.28, Q3 EUR 82.32; Q4 is due 4 Jan 2027. Certificates are bought on the common central platform from February 2027 and priced weekly from 2027. Certificates due are reduced by free allocation, which the CBAM factor phases out: 97.5% in 2026 to none from 2034.",
        "key_dates": [
            {"date": "2023-10-01", "done": True, "event": "Transitional (reporting-only) period starts"},
            {"date": "2026-01-01", "done": True, "event": "Definitive regime applies"},
            {"date": "2026-04-07", "done": True, "event": "Q1 2026 certificate price published: EUR 75.36"},
            {"date": "2026-07-06", "done": True, "event": "Q2 2026 certificate price published: EUR 75.28"},
            {"date": "2026-08-10", "done": True, "event": "Commission updates the default-values workbook after correcting act (EU) 2026/1740"},
            {"date": "2026-10-05", "done": True, "event": "Q3 2026 certificate price published: EUR 82.32"},
            {"date": "2027-01-04", "done": False, "event": "Q4 2026 certificate price due"},
            {"date": "2027-02", "done": False, "event": "All CBAM certificates purchased on the common central platform; weekly prices from 2027"},
        ],
        "key_values": {
            "certificate_price_eur_t_2026": {"Q1": 75.36, "Q2": 75.28, "Q3": 82.32},
            "cbam_factor_on_free_allocation": {"2025": 1.0, "2026": 0.975, "2027": 0.95, "2028": 0.90, "2029": 0.775, "2030": 0.515, "2031": 0.39, "2032": 0.265, "2033": 0.14, "2034": 0.0},
            "cbam_factor_source": "Directive 2003/87/EC Art. 10a(1a) (consolidated 1 Mar 2024), read on EUR-Lex; Implementing Regulation (EU) 2025/2620 applies it to the free-allocation adjustment",
            "authorised_declarant_threshold_t": 50,
        },
        "engine": {"state": "matches", "holds": "Certificate share = 1 minus the CBAM factor (2.5% in 2026 to 100% from 2034); 2026 price = mean of the three published quarters",
                   "where": "climate/cbam.py, finance/policy.py (EU_CBAM_FREE_SHARE)",
                   "note": "Corrected on 6 Nov 2026: climate/cbam.py had assumed 100% from 2026 and a EUR 60 price, which overstated the 2026 bill about forty-fold. Still not loaded: the Commission's default values (2025/2621, corrected by 2026/1740) and the free-allocation benchmarks (2025/2620); embedded intensities remain screening proxies. Prices from 2027 are a scenario path."},
        "not_assessed": ["the Commission proposal COM(2025) 989 to extend CBAM to downstream goods (a proposal; adoption status not checked)", "weekly prices from 2027"],
        "monitors": {
            "cbam_price": {"edition": [3], "label": "3 of 4 quarterly CBAM certificate prices published for 2026 (latest Q3: EUR 82.32)", "date": "2026-10-05"},
            "cbam_law": {"edition": ["2026-08-10", 14], "label": "CBAM legislation page: default values of 10 Aug 2026, 14 implementing/delegated acts listed", "date": "2026-08-10"},
        },
        "sources": [
            {"title": "European Commission: CBAM", "url": "https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en"},
            {"title": "European Commission: Price of CBAM certificates", "url": "https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism/price-cbam-certificates_en"},
            {"title": "European Commission: CBAM legislation and guidance", "url": "https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism/cbam-legislation-and-guidance_en"},
            {"title": "European Commission: CBAM definitive regime", "url": "https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism/cbam-definitive-regime_en"},
            {"title": "EUR-Lex: Directive 2003/87/EC consolidated (CBAM factor, Art. 10a(1a))", "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02003L0087-20240301"},
            {"title": "EUR-Lex: Implementing Regulation (EU) 2025/2620", "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32025R2620"},
        ],
    },
    {
        "id": "pcaf", "name": "PCAF Global GHG Accounting and Reporting Standard", "body": "Partnership for Carbon Accounting Financials",
        "covers": "Part A Financed Emissions (3rd edition, 2025), Part B Facilitated Emissions (December 2023), Part C Insurance-Associated Emissions (2nd edition, 2025)",
        "status": "newer_edition_published",
        "headline": "Part A is now in its 3rd edition (2025; the PDF's file name is dated 15 Jan 2026), replacing the 2nd edition (2022). Part C has a 2nd edition (2025). PCAF notes that GHG Protocol has closed its Built-on review service, so the additions since the first edition have not been reviewed by GHG Protocol.",
        "key_dates": [
            {"date": "2019", "done": True, "event": "Part A 1st edition"},
            {"date": "2022", "done": True, "event": "Part A 2nd edition; Part C 1st edition (November 2022)"},
            {"date": "2023-12", "done": True, "event": "Part B Facilitated Emissions published"},
            {"date": "2025-12", "done": True, "event": "Part A 3rd edition and Part C 2nd edition published (the December 2025 edition)"},
        ],
        "engine": {"state": "behind", "holds": "Part A 2nd edition (2022): attribution factors for listed equity and bonds, business loans, project finance, real estate, mortgages and vehicle loans", "where": "regulatory/pcaf.py",
                   "note": "Not updated to the 3rd edition. I have not read the 3rd edition, so what changed (new asset classes, data-quality scoring, attribution) is not assessed."},
        "not_assessed": ["what the 3rd edition changes", "Part B and Part C are not implemented in the engine"],
        "monitors": {"pcaf": {"edition": [2, 2022], "label": "PCAF Part A 2nd edition (2022)", "date": "2022-12-01"}},
        "sources": [{"title": "PCAF: The Standard", "url": "https://carbonaccountingfinancials.com/en/standard"}],
    },
]

# ── checks ───────────────────────────────────────────────────────────────────

_DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?|-Q[1-4])?$")


def _end(d: str) -> str:
    """Latest day a (possibly partial) date can mean, as YYYY-MM-DD."""
    y = d[:4]
    if re.fullmatch(r"\d{4}", d):
        return f"{y}-12-31"
    if "-Q" in d:
        return {"1": f"{y}-03-31", "2": f"{y}-06-30", "3": f"{y}-09-30", "4": f"{y}-12-31"}[d[-1]]
    if re.fullmatch(r"\d{4}-\d{2}", d):
        return f"{y}-{d[5:7]}-31"
    return d


def _check(items: list) -> None:
    ids = [i["id"] for i in items]
    assert len(set(ids)) == len(ids), "duplicate ids"
    for i in items:
        assert i["status"] in STATUS, f"{i['id']}: unknown status"
        assert i["engine"]["state"] in ENGINE_STATE, f"{i['id']}: unknown engine state"
        assert i["sources"] and all(s["url"].startswith("https://") for s in i["sources"]), f"{i['id']}: needs primary source URLs"
        assert i["headline"].strip() and i["not_assessed"], f"{i['id']}: headline / not_assessed missing"
        assert i["monitors"], f"{i['id']}: needs a monitor fingerprint"
        for d in i["key_dates"]:
            assert _DATE.match(d["date"]), f"{i['id']}: bad date {d['date']}"
            if d["done"]:
                assert _end(d["date"]) <= _end(AS_OF) or d["date"] <= AS_OF, f"{i['id']}: {d['date']} marked done but is after {AS_OF}"
            else:
                assert _end(d["date"]) > AS_OF, f"{i['id']}: {d['date']} marked upcoming but is not after {AS_OF}"
        done = [d["date"] for d in i["key_dates"] if d["done"]]
        assert done == sorted(done), f"{i['id']}: done dates out of order"
    # the CBAM factor schedule must equal the one the finance module uses, and only fall
    cb = next(i for i in items if i["id"] == "cbam")["key_values"]["cbam_factor_on_free_allocation"]
    vals = [cb[str(y)] for y in range(2025, 2035)]
    assert vals == sorted(vals, reverse=True) and vals[0] == 1.0 and vals[-1] == 0.0, "CBAM factor must fall from 100% to 0"
    sys.path.insert(0, str(ROOT / "src"))
    from cri.finance.policy import EU_CBAM_FREE_SHARE
    assert {int(k): v for k, v in cb.items()} == EU_CBAM_FREE_SHARE, "CBAM factor differs from finance/policy.py EU_CBAM_FREE_SHARE"
    prices = next(i for i in items if i["id"] == "cbam")["key_values"]["certificate_price_eur_t_2026"]
    assert len(prices) == next(i for i in items if i["id"] == "cbam")["monitors"]["cbam_price"]["edition"][0], "price count must match the cbam_price fingerprint"


def build() -> dict:
    _check(ITEMS)
    return {"as_of": AS_OF, "status_vocabulary": STATUS,
            "note": "Statuses read from each publisher's own page on 6 Nov 2026. Paywalled standards (ISO) are tracked by edition, stage and dates only. Items under not_assessed were not read from a primary page.",
            "items": ITEMS}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT, f"({len(ITEMS)} items)")
