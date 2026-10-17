# Twin Bench page

One page shell, one module per twin (incubator, battery, UAV). Each module brings its 3D asset, readings, curves and
what-if controls; the shell, event log and six-element panel are shared. Everything runs in the browser.

Published by `.github/workflows/pages.yml` on every push to `main` that touches `web/`.

## Files
| File | What | Edit by hand? |
|---|---|---|
| `page.html` | the page (a fragment without `<head>`) | yes — the only page source |
| `build.py` | wraps `page.html` into `index.html` | yes |
| `index.html` | built page; open it in a browser | no (git-ignored) |
| `incubator_core.js` | incubator model, Kalman filter, controller, simulated box; ported from `twins/incubator/idt` | yes |
| `battery_core.js` | retirement decision; ported from `twins/battery/bdt` | yes |
| `uav_core.js` | Bayes filter, policy, closed-loop asset; ported from `twins/uav/udt` | yes |
| `battery_cells.js` | measured capacity and twin states of 3 cells, from `twins/battery/scripts/export_web.py` | no |
| `uav_model.js` | strain table, transition model, planner reward (authors' data, BSD 3-Clause), from `twins/uav/scripts/export_web.py` | no |
| `replay_2023.js` | the real incubator experiment, from `twins/incubator/scripts/export_reference.py`; loaded only when the page is opened locally | no (git-ignored, never published: the data have no license) |
| `checks/check_<twin>.js` | compares the page's JavaScript with the Python reference | yes |

## Build and check
```bash
python web/build.py                                                      # -> web/index.html
node web/checks/check_incubator.js twins/incubator/results web           # WEB CHECK OK
node web/checks/check_battery.js twins/battery/results web               # BATTERY WEB CHECK OK
node web/checks/check_uav.js twins/uav/results web                       # UAV WEB CHECK OK
```
The incubator check needs `python twins/incubator/scripts/export_reference.py --replay-only` first (local data).

Python is the reference and the JavaScript is checked against it: incubator max difference ~1e-12 °C; battery
615/615 what-if retirement cycles identical; UAV 120/120 steps of maneuvers and policies identical, estimates within ~5e-13.
