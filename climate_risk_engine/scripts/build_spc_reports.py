"""Build src/cri/data/spc_reports_1996_2024.npz from the NOAA Storm Prediction Center public report files.
  curl -O https://www.spc.noaa.gov/wcm/data/1955-2024_hail.csv.zip ; unzip it
  curl -O https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv
  python scripts/build_spc_reports.py <hail.csv> <tornadoes.csv> src/cri/data/spc_reports_1996_2024.npz
Kept: 1996-2024 (Doppler-radar era), hail of 1.0 inch or more (the severe threshold since 2010; earlier reports of 0.75-0.99 inch are dropped so the
series is consistent), every tornado with its F/EF rating. Start point of each report only."""
import sys

import numpy as np
import pandas as pd

hail = pd.read_csv(sys.argv[1])
tor = pd.read_csv(sys.argv[2])
Y0, Y1 = 1996, 2024
h = hail[(hail.yr >= Y0) & (hail.yr <= Y1) & (hail.mag >= 1.0) & (hail.slat != 0) & (hail.slon != 0)]
t = tor[(tor.yr >= Y0) & (tor.yr <= Y1) & (tor.slat != 0) & (tor.slon != 0)]
day = lambda d: (pd.to_datetime(d["date"]).astype("int64") // 86_400_000_000_000).astype("int32").to_numpy()   # noqa: E731
np.savez_compressed(sys.argv[3],
                    hail_day=day(h), hail_lat=h.slat.to_numpy("float32"), hail_lon=h.slon.to_numpy("float32"), hail_in=h.mag.to_numpy("float32"),
                    tor_day=day(t), tor_lat=t.slat.to_numpy("float32"), tor_lon=t.slon.to_numpy("float32"), tor_mag=t.mag.to_numpy("int8"),
                    y0=Y0, y1=Y1)
print(len(h), len(t))
