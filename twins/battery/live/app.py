"""Local web page for the live battery twin: what is coming in, what the twin believes,
what it decided, and what-if questions. Run: streamlit run live/app.py
"""
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bdt.pf import BatteryTwin, power_law  # noqa: E402
from bdt.decision import decide_online  # noqa: E402
import store  # noqa: E402

st.set_page_config(page_title="Battery Digital Twin", layout="wide")
con = store.connect()


def q(sql, *args):
    return pd.read_sql_query(sql, con, params=args)


st.title("Battery digital twin (live)")
st.caption("Physical side: measured cycles of one LFP cell (Severson et al. 2019) replayed over MQTT. "
           "Digital side: particle filter on a power-law fade model, updated once per cycle.")

cells = q("SELECT cell FROM meta")["cell"].tolist()
if not cells:
    st.info("Waiting for data. Start the replayer (see README).")
    time.sleep(1)
    st.rerun()
cell = st.sidebar.selectbox("Cell", cells)


@st.fragment(run_every=1.0)
def live_view():
    meas = q("SELECT cycle, q FROM measurements WHERE cell=? ORDER BY cycle", cell)
    state = q("SELECT * FROM state WHERE cell=? ORDER BY cycle", cell)
    dec = q("SELECT * FROM decisions WHERE cell=?", cell)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest cycle", int(meas.cycle.iloc[-1]) if len(meas) else "-")
    c2.metric("Capacity (normalized)", f"{meas.q.iloc[-1]:.3f}" if len(meas) else "-")
    if len(state):
        s = state.iloc[-1]
        c3.metric("Predicted end of life (cycle)", f"{s.eol_pred:.0f}",
                  help="Cycle where predicted capacity reaches 50%")
        c4.metric("Remaining life, 90% band", f"{s.rul_lo:.0f} – {s.rul_hi:.0f} cycles")
    else:
        c3.metric("Predicted end of life", "updating starts at 98.5%")

    if len(dec):
        d = dec.iloc[0]
        st.success(f"Twin decision (made at cycle {d.cycle_decided}): retire at cycle {d.x_twin}. "
                   f"Sent back to the asset over MQTT.")
    else:
        st.info("Decision is made once capacity first drops to 95%.")

    layers = [alt.Chart(meas).mark_line(color="#333").encode(
        x=alt.X("cycle:Q", title="Cycle"), y=alt.Y("q:Q", title="Normalized capacity",
                                                    scale=alt.Scale(domain=[0.45, 1.02])))]
    if len(state):
        s = state.iloc[-1]
        ks = np.arange(int(s.cycle), int(min(s.eol_pred, 10000)) + 1)
        pred = pd.DataFrame(dict(cycle=ks, q=power_law(ks, s.log10_a, s.b)))
        layers.append(alt.Chart(pred).mark_line(color="#2E86C1", strokeDash=[6, 4]).encode(x="cycle:Q", y="q:Q"))
        band = pd.DataFrame(dict(lo=[s.cycle + s.rul_lo], hi=[s.cycle + s.rul_hi]))
        layers.append(alt.Chart(band).mark_rect(opacity=0.15, color="#2E86C1").encode(x="lo:Q", x2="hi:Q"))
    rules = pd.DataFrame(dict(y=[0.95, 0.5], label=["decision trigger 95%", "end of life 50%"]))
    layers.append(alt.Chart(rules).mark_rule(strokeDash=[2, 2], color="#999").encode(y="y:Q"))
    if len(dec):
        layers.append(alt.Chart(pd.DataFrame(dict(x=[dec.iloc[0].x_twin]))).mark_rule(color="#E74C3C").encode(x="x:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=360), use_container_width=True)
    st.caption("Black: measured. Blue dashed: twin prediction. Blue band: 90% end-of-life range. Red: retirement decision.")


live_view()

st.header("What if")
dec = q("SELECT * FROM decisions WHERE cell=?", cell)
if not len(dec):
    # The live view above refreshes itself; this section waits for the decision, then appears.
    st.write("Appears once the twin has made its decision (capacity first at 95%).")
    time.sleep(1)
    st.rerun()
state = q("SELECT * FROM state WHERE cell=? ORDER BY cycle", cell)
meta = q("SELECT * FROM meta WHERE cell=?", cell).iloc[0]
ref = int(dec.iloc[0].cycle_decided)
s = state[state.cycle == ref].iloc[0]
hist = q("SELECT q FROM measurements WHERE cell=? AND cycle<=? ORDER BY cycle", cell, ref)["q"].to_numpy()
st.caption(f"Questions are asked at cycle {ref}, when the twin made its decision, using only data up to then.")

w = st.slider("Weight on total energy delivered (rest: time between charges)", 0.0, 1.0, 0.5, 0.05)
r = decide_online(hist, meta.q0, s.log10_a, s.b, s.eol_pred, weights=(w, 1 - w))
x = st.slider("Retire at cycle", int(r["xs"][0]), int(r["xs"][-1]), int(r["x_twin"]))
u_at = float(r["utility"][x - r["xs"][0]])
a, b = st.columns(2)
a.metric("Twin's best retirement cycle", r["x_twin"])
b.metric("Utility if retired at chosen cycle", f"{u_at:.3f}", f"{u_at - np.nanmax(r['utility']):+.3f} vs best")
curve = pd.DataFrame(dict(cycle=r["xs"], utility=r["utility"]))
st.altair_chart(alt.Chart(curve).mark_line().encode(x=alt.X("cycle:Q", title="Retirement cycle"),
                                                    y=alt.Y("utility:Q", title="Utility", scale=alt.Scale(zero=False)))
                + alt.Chart(pd.DataFrame(dict(x=[x]))).mark_rule(color="#E74C3C").encode(x="x:Q"),
                use_container_width=True)

st.subheader("What if the sensor were noisier?")
noise = st.select_slider("Extra measurement noise (std, normalized capacity)", [0.0, 0.002, 0.005, 0.01, 0.02], 0.005)
st.caption("Adds one random draw of Gaussian noise to the measured history and re-runs the twin; "
           "a different draw gives a different answer.")
if st.button("Re-run the twin on the same history with this noise"):
    rng = np.random.RandomState(1)
    noisy = hist + rng.randn(len(hist)) * noise
    start = int(np.where(noisy <= 0.985)[0][0])
    tw = BatteryTwin(rng=np.random.RandomState(0))
    for i in range(start, len(noisy)):
        out = tw.step(i + 2, noisy[i])
    r2 = decide_online(noisy, meta.q0, out["log10_a"], out["b"], out["eol_pred"], weights=(w, 1 - w))
    c1, c2 = st.columns(2)
    c1.metric("Predicted end of life", f"{out['eol_pred']:.0f}", f"{out['eol_pred'] - s.eol_pred:+.0f} vs clean")
    c2.metric("Best retirement cycle", r2["x_twin"], f"{r2['x_twin'] - r['x_twin']:+d} vs clean")
