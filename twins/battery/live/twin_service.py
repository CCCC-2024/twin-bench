"""Digital side: receives measurements over MQTT, updates the twin one cycle at a time,
stores everything in SQLite, and sends the retirement decision back (digital -> physical).

Update starts at the first-prediction time (normalized capacity <= 0.985), as in the original
code; earlier cycles are stored and shown but not filtered. The decision is made once, when
capacity first drops to 0.95, using only information available at that cycle.
"""
import json
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt

from twins.battery.data import FPT_THRESHOLD
from twins.battery.decision import decide_online
from twins.battery.live import store
from twins.battery.pf import ParticleFilter

DECIDE_AT = 0.95
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

con = store.connect(reset=True)
twins, hist, meta, decided = {}, {}, {}, set()


def on_meta(c, u, m):
    d = json.loads(m.payload)
    meta[d["cell"]] = d
    con.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?,?)", (d["cell"], d["q0"], None, d["last_measured"]))
    con.commit()


def on_capacity(c, u, m):
    d = json.loads(m.payload)
    name, cycle, q = d["cell"], d["cycle"], d["q"]
    hist.setdefault(name, []).append(q)
    con.execute("INSERT INTO measurements VALUES (?,?,?,?,?)", (name, cycle, q, d["cap_ah"], d["t"]))
    if name not in twins and q <= FPT_THRESHOLD:              # first-prediction time reached
        twins[name] = ParticleFilter(rng=np.random.RandomState(SEED))
        con.execute("UPDATE meta SET k_fpt=? WHERE cell=?", (cycle, name))
    if name in twins:
        s = twins[name].step(cycle, q)
        con.execute("INSERT INTO state VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (name, cycle, s["log10_a"], s["b"], s["log10_a_std"], s["b_std"],
                     s["eol_pred"], s["rul_lo"], s["rul_hi"], time.time()))
        c.publish(f"twin/battery/{name}/state", json.dumps({k: float(v) for k, v in s.items()}))
        if name not in decided and q <= DECIDE_AT:
            q0 = meta.get(name, {}).get("q0", d["cap_ah"] / q)
            dec = decide_online(np.array(hist[name]), q0, s["log10_a"], s["b"], s["eol_pred"])
            decided.add(name)
            con.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                        (name, cycle, dec["x_twin"], dec["horizon"], time.time()))
            c.publish(f"twin/battery/{name}/decision", json.dumps(
                dict(cycle_decided=cycle, x_twin=dec["x_twin"], horizon=dec["horizon"])), qos=1)
            print(f"[twin] {name}: at cycle {cycle} -> retire at {dec['x_twin']} "
                  f"(predicted EOL {dec['horizon']})", flush=True)
    con.commit()


cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="twin-service")
cli.message_callback_add("battery/+/meta", on_meta)
cli.message_callback_add("battery/+/capacity", on_capacity)
cli.connect("127.0.0.1", 1883)
cli.subscribe([("battery/+/meta", 1), ("battery/+/capacity", 1)])
print(f"[twin] listening (seed {SEED})", flush=True)
cli.loop_forever()
