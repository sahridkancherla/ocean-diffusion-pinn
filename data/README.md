# Data

Observations from the NOAA **Kuroshio Extension Observatory (KEO)** surface mooring
at 32.3° N, 144.6° E in the western North Pacific, operated by the
[Ocean Climate Stations (OCS)](https://www.pmel.noaa.gov/ocs/KEO) project at NOAA PMEL.
The files were retrieved from the OCS [mooring data portal](https://www.pmel.noaa.gov/ocs/data/disdel/)
on 29 November 2025.

| File | Contents | Used by the model |
|---|---|---|
| `temp.cdf.zip` | Subsurface temperature `T_20` (°C) and pressure `P_1` (dbar), with quality (`QT_5020`, `QP_5001`) and source (`ST_6020`, `SP_6001`) flags | yes |
| `sst.cdf` | Sea surface temperature `T_25` (°C) at 1 m, with quality and source flags | no |
| `pos.cdf` | Daily GPS position of the buoy (`LAT_500`, `LON_502`) | no |

`temp.cdf.zip` holds a single NetCDF file of about 140 MB, zipped to stay under
GitHub's 100 MB file limit. The code extracts it to `data/temp.cdf` (ignored by git)
on first use.

All three files are NetCDF-3 in the EPIC conventions used by PMEL, with missing samples
stored as `1e35`. The temperature and SST records run from 2015-01-01 00:00 to
2020-05-19 03:00 at 10-minute intervals (282,979 samples).

## Depth levels

Share of temperature samples missing at each nominal depth:

| Depth (m) | 1 | 5 | 10 | 15 | 20 | 25 | 35 | 40 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Missing (%) | 3 | 7 | 46 | 26 | 47 | 32 | 5 | 9 | 19 | 19 | 25 |

| Depth (m) | 125 | 150 | 175 | 225 | 275 | 325 | 375 | 425 | 475 | 525 |
|---|---|---|---|---|---|---|---|---|---|---|
| Missing (%) | 13 | 16 | 3 | 11 | 15 | 20 | 12 | 26 | 31 | 8 |

Pressure is missing at the same times as temperature, except at 1 m, which has no
pressure sensor.

## Preprocessing

Implemented in [`src/ocean_pinn/data.py`](../src/ocean_pinn/data.py):

1. The 1 m sensor, which has no pressure record, is assigned a pressure of 1 dbar.
2. The 25 m and 150 m levels, whose records both begin with a gap, are dropped,
   leaving 19 levels.
3. Missing samples are replaced by the last valid sample from the same sensor.
4. Pressure is converted to height $z$ (m, negative below the surface) with TEOS-10
   (`gsw.z_from_p`) at 32.3° N. PMEL advises against using the nominal depths because
   the mooring line moves.
5. The (time, level) grid is flattened level by level into 5,376,601 points and
   normalised as described in the [main README](../README.md#method).
