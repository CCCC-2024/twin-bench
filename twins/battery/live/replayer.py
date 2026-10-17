"""Physical side: replays one cell's *measured* cycles over MQTT, one message per cycle.

The authors' 3000-cycle linear extrapolation is not sent: it is not a measurement.
Usage: python live/replayer.py --cell 27 --rate 20
"""
import argparse
import json
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bdt.data import load_cells, split  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cell", type=int, default=27, help="test-cell index (27 = authors' example, Primary Test Cell 28)")
ap.add_argument("--rate", type=float, default=20, help="cycles per second")
ap.add_argument("--host", default="127.0.0.1")
args = ap.parse_args()

cell = split(load_cells())[1][args.cell]
name = cell["barcode"]
cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"replayer-{name}")
cli.connect(args.host, 1883)
cli.loop_start()

decision = {}
def on_decision(c, u, m):  # digital -> physical: the asset receives the twin's decision
    decision.update(json.loads(m.payload))
    print(f"[asset] twin says: retire at cycle {decision['x_twin']}", flush=True)
cli.message_callback_add(f"twin/battery/{name}/decision", on_decision)
cli.subscribe(f"twin/battery/{name}/decision")

meta = dict(cell=name, q0=cell["q0"], last_measured=cell["last_measured_cycle"])
cli.publish(f"battery/{name}/meta", json.dumps(meta), qos=1, retain=True).wait_for_publish()
n = cell["n_measured"]
print(f"[asset] {name}: sending {n} measured cycles at {args.rate}/s", flush=True)
for i in range(n):
    msg = dict(cell=name, cycle=int(cell["cycles"][i]), q=float(cell["q"][i]),
               cap_ah=float(cell["cap"][i]), t=time.time())
    cli.publish(f"battery/{name}/capacity", json.dumps(msg), qos=1).wait_for_publish()
    time.sleep(1 / args.rate)
cli.publish(f"battery/{name}/end", "{}", qos=1).wait_for_publish()
print("[asset] done", flush=True)
time.sleep(1)
cli.loop_stop()
