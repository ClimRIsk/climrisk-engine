# Desktop app check, 16 Oct 2026 (numbers, physics, UI)

Checked in the running app: dashboard, Asset Intelligence 3D (Houston, Kedarnath), Risk Analyst, Carbon Accounting worked example, Real Estate worked example, chat.

## Found and fixed in the engine (live in the app after Sync-Engine)
1. Kedarnath headline read "0.00% physical damage" while the site analysis rated debris flow HIGH. The headline now names every hazard the loss does not price and says 0.00% means the priced hazards (cyclone, river flood, coastal flood) are low, not that the site is safe.
2. Houston "59% of value" is circular: the asset value ($316k) is imputed from emissions (563 t/yr), so a carbon cost measured against it is nearly fixed. The headline now says when the value is an estimate and that the percentage is indicative only.
3. Risk Analyst started before the satellite analysis finished never picked the findings up. It now refreshes when they arrive.
4. "0.00 Mt CO2e/yr" for 563 t. Emissions now print in t, kt or Mt.
5. "95th percentile $515k, 100.0%" on a $316k asset: the percentage is capped per draw. Labelled "(capped at value)" with an explanation; the scenario table notes that Total loss PV is capped and so can be below "with response" plus "stranded".
6. Chat "Open the Stress Test" built a 3D twin at the last site. It now opens the screen.
7. Autopilot posted a new, nearly identical data-source alert whenever a different source timed out. One alert a day.

## Found and fixed in the website repo (shipped in the 16 Oct rebuild)
- macOS window buttons overlapped the logo and every tool-overlay title.
- "Physical Risk (25 Hz)" used an icon name that does not exist in Phosphor.
- Earlier fixes (hazard badge scope, "no analysis open" message, Stress Test sidebar entry) are in dashboard.html and now in the build.
- A hardcoded owner licence token and a `?unlock=` back door were removed from the app, the pages and the Electron preload. Every build now needs a signed licence.

## Verified consistent
Carbon Accounting worked example: scope totals equal the sum of entities, categories and gases; intensity = (S1+S2 location)/revenue. Real Estate worked example: EUI, area-weighted intensity and total tonnes reconcile.

## Open
- "Defended to 1-in-100/500" comes from FLOPROS, a modelled national/regional design standard, not a survey of the site's levees. The asset screen now says so; it is still doubtful for Kedarnath and Houston.
- The Electron install was incomplete after the disk clean-up (npm's unzip stopped early). `mac-test/Build-App-Mac.command` now extracts with the system tool and refuses to finish if the engine is not bundled into the app. (It first built an app without the engine, because an absolute path was glued onto the relative one; the script now catches this.)
- The model's held-out scores come from template-built rows, so they are optimistic versus free-form questions.
- Clean-machine install test and Windows build: not done.
