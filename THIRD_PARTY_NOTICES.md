# Third-party notices

Twin Bench's own code is under the MIT License (`LICENSE`). Third-party code and data are **not stored in this
repository**; each twin's `fetch_upstream.sh` downloads them at a pinned commit into `twins/<name>/upstream/`, where
they stay under their own licenses. Material derived from them that *is* in the repository or on the published page
is listed below with its license.

| Twin | Source | Commit | License | Derived material in this repository / page |
|---|---|---|---|---|
| battery | [acthelen/battery_digital_twin](https://github.com/acthelen/battery_digital_twin) (Thelen et al. 2023) | `869a1f6` | MIT | `web/battery_cells.js`: measured capacity of 3 cells and twin states; `twins/battery/results/` |
| battery | cell data: Severson et al. 2019, *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391 | as packaged in the repository above | redistributed there under MIT; see the original dataset's terms at https://data.matr.io/1/ | as above |
| uav | [michaelkapteyn/UAV-Digital-Twin](https://github.com/michaelkapteyn/UAV-Digital-Twin) (Kapteyn, Pretorius & Willcox 2021) | `b51e42d` | BSD 3-Clause | `web/uav_model.js`: strain table, transition model and planner reward from `UAVconfig.json`; ports of the planner in `twins/uav/model.py` and `web/uav_core.js` |
| uav | [michaelkapteyn/UAV-Experimental-Calibration](https://github.com/michaelkapteyn/UAV-Experimental-Calibration) | `1dab634` | BSD 3-Clause | none |
| incubator | [INTO-CPS-Association/example_digital-twin_incubator](https://github.com/INTO-CPS-Association/example_digital-twin_incubator) (Feng et al. 2021) | `0a851c0` | **no license file** | none. The data are used locally only; files that reproduce them (`web/incubator_replay.js`, `twins/incubator/results/ref_web_incubator_replay.json`) are git-ignored and never published. The model is our own code written from the paper. |
| tumor | [OncologyModelingGroup/TumorTwin](https://github.com/OncologyModelingGroup/TumorTwin) | `bedf90a` | UT Austin Research License (academic / research use only) | `twins/tumor/results/`: our runs of the model |
| bikeshare | [DanieleGammelli/variational-poisson-rnn](https://github.com/DanieleGammelli/variational-poisson-rnn) (Gammelli et al. 2022) | `abf77f7` | MIT | `twins/bikeshare/results/`: our decisions and daily shortages, and the upstream decision code's answers on the same inputs; `web/bikeshare_model.js`: the twin's learned totals per hour and day type and the Nov–Dec 2018 average demand (aggregates); `twins/bikeshare/rnn.py`: the VP-RNN inference network adapted from `src/algos/vprnn.py` (the trained weights are fetched, never committed) |
| bikeshare | trip data: Citi Bike System Data 2018 (NYC Bike Share), aggregated per station in the repository above | as packaged above | NYCBS Data Use Policy (https://citibikenyc.com/data-sharing-policy): use and distribution inside a product allowed; no redistribution as a stand-alone dataset | aggregates only (see above); the events and hourly counts are never committed or published (`web/bikeshare_replay.js` is built locally and git-ignored) |
| bikeshare | hourly weather, New York 2018 (`data/raw/weather2018_60min.csv`), as packaged in the repository above; the authors do not state its origin | as packaged above | not stated | none; fetched and used locally as an input of the RNN |

The page loads three.js r128 (MIT) from cdnjs/jsDelivr and fonts from Google Fonts (SIL Open Font License) at run time.

## License texts

### battery_digital_twin (MIT)

```
MIT License

Copyright (c) 2022 Adam Thelen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### variational-poisson-rnn (MIT)

```
MIT License

Copyright (c) 2021 Daniele Gammelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### UAV-Digital-Twin and UAV-Experimental-Calibration (BSD 3-Clause)

```
Copyright (c) 2021, Michael Kapteyn
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Massachusetts Institute of Technology nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
