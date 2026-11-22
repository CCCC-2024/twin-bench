"""Events in words, on every twin: a language model extracts the conditions, the twin compares every action.

For each twin, in a few states, four kinds of event (EVENTS below): stated (its numbers are in the text), vague (no
numbers), nothing (it changes nothing the choice depends on) and inexpressible (the twin's what-if has no input for
it). dtcore.agent.advise gives the model's reading to twin.best(**conditions). No record holds these events, so a
choice is scored by the twin's own what-if under the reference conditions written next to each event:
  extra  how much worse the chosen action is there than the twin's best (0 is best; lost riders, minutes outside the
         band, utility, planner value). A vague event has several references, a mild, a middle and a strong reading
         of it, and extra is their mean. Without an action from the model (cannot, or no valid reply) the twin keeps
         its usual choice, so extra is that of "not_told", the twin that does not know the event.
  right  stated: the conditions are the reference's; nothing: they change nothing; inexpressible: the model said
         cannot. A vague event has no right reading: compare its extra with not_told's.
This measures whether words reach the twin as the right inputs, not what the event would do to the real system.

The first event of each kind is "dev": the prompt and the calls of advise were worked out on these. Those numbered 1-3
are "test": written before advise was run on them, and not changed since. Those numbered 11 and up are "test2", a
second held-out set written on 11/22 after the results on "test" had been seen, for anything changed after that
(asking for a range for a vague event): mostly vague events, for the two twins whose decision depends on them, with
more states (EXTRA_STATES) so that each vague event is scored in several. Its references were committed before any
language model was run on it. A vague test2 event is kept only if not knowing it costs something in at least half its
states, judged by the perfect reader alone.

MODEL reference is no language model: it answers with the (middle) reference, so its rows are what a perfect reader
gives. A row is named by the model and how it was run: ", tools" (the calls as tool calls, not as JSON in the
reply), ", 1 reply" (no second reply after an error) and ", reasoning medium" or ", reasoning high" (rows without
it ran with the least: low for gpt-oss, none for Qwen). Writes twins/<name>/results/advise_events.json, adding to it
row by row.

Usage: python tools/advise_events.py MODEL [MODEL ...] [--url URL] [--reasoning none] [--max-tokens N] [--tools] [--turns 1]
                                     [--twins incubator battery ...] [--split dev]
"""
import argparse
import datetime
import inspect
import json
from pathlib import Path

import numpy as np

from dtcore import agent, run

ROOT = Path(__file__).resolve().parents[1]
CANNOT = "cannot"       # reference of an event the what-if has no input for: the right answer is the call cannot
HOURS = dict(from_hour=7, to_hour=10)

# twin -> the number of whatif() a choice is scored by, +1 if lower is better or -1 if higher, and its events:
# name -> (text, reference conditions: one, or for a vague event a mild, a middle and a strong one)
EVENTS = dict(
    bikeshare=("expected_lost", +1, {
        "stated": ("Tomorrow a subway line near the station is closed from 7:00 to 10:00. In those hours about three "
                   "times the usual number of people will rent bikes here; returns stay as usual.", dict(rent_factor=3, **HOURS)),
        "vague": ("A subway line near the station is closed tomorrow morning.",
                  [dict(rent_factor=f, **HOURS) for f in (2, 3, 4)]),
        "nothing": ("The station's docks are repainted tonight; they stay in use.", {}),
        "inexpressible": ("Tomorrow the truck can carry at most 5 bikes.", CANNOT),
        "stated-1": ("A street fair two blocks away runs tomorrow from noon to 6 pm. During the fair, expect rentals and "
                     "returns here at twice the usual level.", dict(rent_factor=2, return_factor=2, from_hour=12, to_hour=18)),
        "stated-2": ("Heavy rain is forecast for all of tomorrow; ridership will be about 40 % of normal, rentals and "
                     "returns alike.", dict(rent_factor=0.4, return_factor=0.4)),
        "stated-3": ("A concert next to the station ends at 10 pm tomorrow; from 22:00 to midnight rentals will be about "
                     "four times the usual.", dict(rent_factor=4, from_hour=22, to_hour=24)),
        "vague-1": ("Tomorrow is a public holiday; far fewer commuters than usual will use the station.",
                    [dict(rent_factor=f, return_factor=f) for f in (0.7, 0.5, 0.3)]),
        "vague-2": ("A big football game at the stadium next to the station ends around 5 pm tomorrow.",
                    [dict(rent_factor=f, from_hour=17, to_hour=19) for f in (2, 3, 4)]),
        "nothing-1": ("The operator's app gets a new logo tomorrow.", {}),
        "nothing-2": ("The station's name sign is replaced tomorrow; the docks stay in use.", {}),
        "inexpressible-1": ("Tomorrow three of the station's docks are broken and cannot take bikes.", CANNOT),
        "inexpressible-2": ("From tomorrow a lost rental costs the operator twice as much as a lost return.", CANNOT),
        # test2. Vague events: references are rentals and returns times a factor over the hours the event covers, mild
        # / middle / strong. Factors for weather follow the usual finding that rain or snow cuts bike-share trips by
        # roughly half or more; the others are my estimates, stated before any model was run.
        "vague-11": ("A transit strike shuts down all subway service in the city tomorrow.",
                     [dict(rent_factor=f, return_factor=f) for f in (1.5, 2, 3)]),
        "vague-12": ("A snowstorm will hit the city early tomorrow morning and last all day.",
                     [dict(rent_factor=f, return_factor=f) for f in (0.6, 0.4, 0.2)]),
        "vague-13": ("Tomorrow morning the subway station next to the bike station is closed for emergency repairs "
                     "until noon.", [dict(rent_factor=f, from_hour=7, to_hour=12) for f in (2, 3, 4)]),
        "vague-14": ("A marathon closes the avenue in front of the station tomorrow morning; the docks can still be "
                     "reached on foot.", [dict(rent_factor=f, return_factor=f, from_hour=6, to_hour=12) for f in (0.6, 0.4, 0.2)]),
        "vague-15": ("Freezing rain is forecast for all of tomorrow afternoon and evening.",
                     [dict(rent_factor=f, return_factor=f, from_hour=12, to_hour=22) for f in (0.6, 0.4, 0.2)]),
        "vague-16": ("A large conference opens tomorrow morning at the convention center next to the station, and many "
                     "attendees are expected to arrive by bike.", [dict(return_factor=f, from_hour=7, to_hour=10) for f in (2, 3, 4)]),
        "vague-17": ("The protected bike lane on the bridge leading to the station reopens tomorrow after months of "
                     "repairs.", [dict(rent_factor=f, return_factor=f) for f in (1.2, 1.4, 1.6)]),
        "vague-18": ("Tomorrow is a city-wide day off for a parade downtown in the afternoon; offices near the station "
                     "are closed.", [dict(rent_factor=f, return_factor=f) for f in (0.8, 0.6, 0.4)]),
        "stated-11": ("Tomorrow from 8:00 to 11:00 returns will be twice the usual; rentals stay as usual.",
                      dict(return_factor=2, from_hour=8, to_hour=11)),
        "nothing-11": ("The map panel at the station is updated tomorrow.", {}),
        "inexpressible-11": ("Tomorrow the station gets ten extra docks.", CANNOT),
        "inexpressible-12": ("Tomorrow the operator can send a second truck at noon.", CANNOT)}),
    incubator=("minutes_outside", +1, {
        "stated": ("The building's heating is off tonight; the room will be at about 12 °C for the next hour.", dict(T_room=12)),
        "vague": ("The building's heating is off tonight and it is freezing outside.", [dict(T_room=t) for t in (16, 12, 8)]),
        "nothing": ("A new label was put on the door of the box.", {}),
        "inexpressible": ("Someone will hold the lid open for five minutes to take samples.", CANNOT),
        "stated-1": ("A heat wave: the air conditioning failed and the room is at 31 °C for the next hour.", dict(T_room=31)),
        "stated-2": ("The lab window was left open; the room has dropped to 59 °F.", dict(T_room=15)),
        "stated-3": ("The room thermostat was reset; the room will hold 18 degrees Celsius from now on.", dict(T_room=18)),
        "vague-1": ("The air conditioning broke on a hot summer afternoon.", [dict(T_room=t) for t in (27, 30, 33)]),
        "vague-2": ("Someone left the lab window open on a cold winter night.", [dict(T_room=t) for t in (18, 14, 10)]),
        "nothing-1": ("The lab's ceiling lights were switched to LED.", {}),
        "nothing-2": ("The incubator was given a new inventory number.", {}),
        "inexpressible-1": ("A faulty supply cuts the heater's power in half.", CANNOT),
        "inexpressible-2": ("A tray of cold samples will be put into the box in ten minutes.", CANNOT),
        # test2. Vague events: references are the room temperature for the next hour, mild / middle / strong. A
        # cold room is held near 4 °C by design; the others are my estimates of an indoor room, stated before any
        # model was run.
        "vague-11": ("The air conditioning in the lab stopped working an hour ago on a hot summer day.",
                     [dict(T_room=t) for t in (27, 30, 33)]),
        "vague-12": ("The lab's heating broke overnight during a cold snap, and the room feels chilly.",
                     [dict(T_room=t) for t in (18, 16, 14)]),
        "vague-13": ("The incubator was moved next to a sunny window this afternoon.", [dict(T_room=t) for t in (26, 29, 32)]),
        "vague-14": ("Someone switched on a space heater right next to the incubator.", [dict(T_room=t) for t in (28, 32, 36)]),
        "vague-15": ("The incubator has been carried into the walk-in cold room.", [dict(T_room=t) for t in (8, 4, 2)]),
        "vague-16": ("The incubator now stands in the warm room used for drying glassware.", [dict(T_room=t) for t in (30, 35, 40)]),
        "vague-17": ("The building's thermostat was turned down for the weekend to save energy.",
                     [dict(T_room=t) for t in (19, 17, 15)]),
        "vague-18": ("It is a freezing winter morning and the ventilation is blowing outside air into the lab.",
                     [dict(T_room=t) for t in (18, 15, 12)]),
        "stated-11": ("The room will be held at 20 °C for the next hour.", dict(T_room=20)),
        "nothing-11": ("The lab got a new coffee machine.", {}),
        "inexpressible-11": ("The heater is being replaced with one twice as powerful.", CANNOT),
        "inexpressible-12": ("From now on the temperature sensor reads half a degree too high.", CANNOT)}),
    battery=("utility", -1, {
        "stated": ("From now on this cell is fast-charged; cells used this way lose capacity about 1.5 times as fast.",
                   dict(fade_factor=1.5)),
        "vague": ("From now on this cell does much harsher duty: fast charging in a hot room.",
                  [dict(fade_factor=f) for f in (1.25, 1.5, 2)]),
        "nothing": ("The cell got a new serial-number sticker.", {}),
        "inexpressible": ("The buyer of retired cells now pays twice as much for each one.", CANNOT),
        "stated-1": ("From now on the cell is charged slowly and kept cool; cells used this way lose capacity about half "
                     "as fast.", dict(fade_factor=0.5)),
        "stated-2": ("The cell moves to a delivery van that fast-charges twice a day; expect capacity loss at twice the "
                     "present rate.", dict(fade_factor=2)),
        "stated-3": ("The pack's cooling was repaired; from now on the cell should fade 20 % more slowly.", dict(fade_factor=0.8)),
        "vague-1": ("From now on the cell works in a rooftop unit in the desert, hot most of the day.",
                    [dict(fade_factor=f) for f in (1.3, 1.8, 2.5)]),
        "vague-2": ("The cell is moved to gentle duty: shallow cycles in an air-conditioned room.",
                    [dict(fade_factor=f) for f in (0.8, 0.6, 0.4)]),
        "nothing-1": ("The cell's test bench was moved to another room at the same temperature.", {}),
        "nothing-2": ("A different technician now logs the cell's data.", {}),
        "inexpressible-1": ("A second-life buyer will only accept cells retired above 85 % of their first capacity.", CANNOT),
        "inexpressible-2": ("From now on, total energy delivered counts three times as much as run time per charge.", CANNOT),
        "stated-11": ("From now on the cell loses capacity 1.2 times as fast as it does now.", dict(fade_factor=1.2)),
        "nothing-11": ("The cell's log files were moved to a new server.", {}),
        "inexpressible-11": ("The cell will sit unused in storage for six months before its next cycle.", CANNOT),
        "inexpressible-12": ("The cell's internal resistance has just doubled.", CANNOT)}),
    uav=("value", -1, {
        "stated": ("Strong gusts are forecast on the route; wing damage will grow about twice as often as usual.",
                   dict(growth_factor=2)),
        "vague": ("Strong gusts are forecast on the route.", [dict(growth_factor=f) for f in (1.5, 2, 3)]),
        "nothing": ("The drone was repainted grey.", {}),
        "inexpressible": ("The mission is urgent today: arriving fast matters far more than usual.", CANNOT),
        "stated-1": ("Calm air all along the route today; wing damage will grow about half as often as usual.",
                     dict(growth_factor=0.5)),
        "stated-2": ("The drone carries a heavy payload; damage growth is expected to be three times as frequent.",
                     dict(growth_factor=3)),
        "stated-3": ("Turbulence over the ridge: expect damage to grow 1.5 times as often as usual.", dict(growth_factor=1.5)),
        "vague-1": ("A storm front is crossing the route, with severe turbulence.", [dict(growth_factor=f) for f in (1.5, 2, 3)]),
        "vague-2": ("Perfectly calm, cool air is forecast for the whole route.", [dict(growth_factor=f) for f in (0.8, 0.5, 0.3)]),
        "nothing-1": ("The operator's logo was added to the drone's tail.", {}),
        "nothing-2": ("The pilot's console got a new colour theme.", {}),
        "inexpressible-1": ("Region 2 of the wing was just repaired to like-new.", CANNOT),
        "inexpressible-2": ("Airspace rules allow only gentle 2g turns today.", CANNOT),
        "stated-11": ("Damage will grow 2.5 times as often as usual from now on.", dict(growth_factor=2.5)),
        "nothing-11": ("The ground station's software was updated.", {}),
        "inexpressible-11": ("Fuel is low: the drone must land within ten turns.", CANNOT),
        "inexpressible-12": ("Each 3g turn now saves twice as much time as before.", CANNOT)}),
)


EXTRA_STATES = dict(bikeshare=("2018-11-07", "2018-11-14", "2018-12-05"),       # nights, for test2 events only
                    incubator=((37.0, 18.0), (37.0, 26.0), (36.5, 22.0)))          # (set point, room °C)


def states(name, extra=False):
    """(label, twin, is_extra) for one twin, each twin running and about to decide; with extra, EXTRA_STATES too."""
    if name == "bikeshare":                 # 23:00 on three nights, the rate model remembering the last 21 days
        from twins.bikeshare import HistoricalRates, ReplayStation, StationTwin, data
        jan1, C = datetime.date(2018, 1, 1), data.capacity()
        days = [(d, False) for d in ("2018-11-01", "2018-11-25", "2018-12-19")] + \
            [(d, True) for d in EXTRA_STATES["bikeshare"] if extra]
        for day, more in sorted(days):
            day = datetime.date.fromisoformat(day)
            twin = StationTwin(C, HistoricalRates(window_days=21))
            run(ReplayStation(C, data.events(), jan1, 365), twin, max_steps=(day - jan1).days * 24)
            yield f"night before {day}", twin, more
    elif name == "incubator":               # the simulated box after an hour at a set point
        from twins.incubator import Controller, IncubatorTwin, SimBox
        for (desired, room), more in [((37.0, 22.0), False)] + [(s, True) for s in EXTRA_STATES["incubator"] if extra]:
            box = SimBox(T_room=room, controller=Controller(desired=desired, lower_bound=1.0))
            twin = IncubatorTwin(controller=box.ctrl, band=(36.0, 38.0))
            run(box, twin, max_steps=1200)
            yield "an hour in, room at 22 °C" if not more else f"an hour in at {desired:g} °C, room at {room:g} °C", twin, more
    elif name == "battery":                 # three test cells when they first reach 95 % capacity
        from twins.battery import BatteryTwin, CellReplay, data
        for cell in data.split(data.load_cells())[1][:3]:
            twin = BatteryTwin(cell["q0"])
            n = int(np.where(cell["q"][:cell["n_measured"]] <= twin.decide_at)[0][0]) + 1
            run(CellReplay(cell), twin, max_steps=n)
            yield f"cell {cell['barcode']} at cycle {twin.last['cycle']}", twin, False
    elif name == "uav":                     # the simulated wing with region 1 damaged 0, 2 and 3 times (20 % each)
        from twins.uav import ControlledUAV, UAVModel, UAVTwin
        m = UAVModel()
        for hits in (0, 2, 3):
            uav, twin = ControlledUAV(m, growth=np.ones((100000, 2))), UAVTwin(m)     # damage only on purpose
            for _ in range(hits):
                uav.damage(0)
            run(uav, twin, max_steps=10)
            yield f"region 1 at {20 * hits} %", twin, False


def reference_model(reference):
    """A perfect reader: the call the reference means (the middle one of a vague event's)."""
    call = dict(name="cannot", arguments=dict(why="not an input of the what-if")) if reference == CANNOT else \
        dict(name="conditions", arguments=reference[len(reference) // 2])
    return lambda messages, tools=None: json.dumps(call)


def extra(twin, key, sign, action, references):
    """How much worse `action` is than the twin's best under the reference conditions, averaged over them."""
    take = lambda a: {k: a[k] for k in twin.actions}
    return round(float(np.mean([sign * (twin.whatif(**take(action), **r)[key] - twin.whatif(**take(twin.best(**r)), **r)[key])
                                for r in references])), 4) + 0.0


def same(twin, conditions, reference):
    """Whether two sets of conditions are the same input of best(): one left out is at its default."""
    full = lambda c: {k: c.get(k, p.default) for k, p in inspect.signature(twin.best).parameters.items()}
    return conditions is not None and set(conditions) <= set(twin.conditions) and full(conditions) == full(reference)


parser = argparse.ArgumentParser(description="Events in words on every twin: the model extracts, the twin compares.")
parser.add_argument("models", nargs="+", help='language models, as the server names them; "reference" is a perfect reader')
parser.add_argument("--url", default="http://localhost:11434/v1", help="the server's OpenAI chat API (default: Ollama)")
parser.add_argument("--reasoning", help="reasoning_effort for the models, e.g. none or low (default: theirs)")
parser.add_argument("--max-tokens", type=int, default=4096, help="cap on each reply, thinking included (default 4096)")
parser.add_argument("--tools", action="store_true", help="the calls as tool calls, not as JSON in the reply")
parser.add_argument("--turns", type=int, default=2, help="replies the model may give to one event (default 2)")
parser.add_argument("--twins", nargs="+", default=list(EVENTS), choices=list(EVENTS))
parser.add_argument("--split", choices=("dev", "test", "test2"), help="only these events (default: all)")
args = parser.parse_args()
how = (", tools" if args.tools else "") + (", 1 reply" if args.turns == 1 else "") + \
    (f", reasoning {args.reasoning}" if args.reasoning in ("medium", "high") else "")
rows_of = {m: m if m == "reference" else m + how for m in args.models}
split = lambda name: "dev" if not name[-1].isdigit() else "test2" if int(name.split("-")[1]) >= 11 else "test"
kinds = ("stated", "vague", "nothing", "inexpressible")
total = {}                                                      # (row, split, kind) -> the cells of every twin

for twin_name in args.twins:
    key, sign, events = EVENTS[twin_name]
    events = {n: (t, r if r == CANNOT or isinstance(r, list) else [r]) for n, (t, r) in events.items()}
    out = ROOT / "twins" / twin_name / "results" / "advise_events.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    res.update(scored_by=key, events={n: dict(split=split(n), text=t, reference=r) for n, (t, r) in events.items()})
    states_ = res.setdefault("states", {})
    for label, twin, more in states(twin_name, extra=args.split in (None, "test2")):
        for name, (text, reference) in events.items():
            if args.split not in (None, split(name)) or (more and split(name) != "test2"):
                continue
            row = states_.setdefault(label, {}).setdefault(name, {})
            if reference != CANNOT:
                usual = twin.best()
                row["not_told"] = dict(usual, extra=extra(twin, key, sign, usual, reference))
                if len(reference) == 1:
                    row["best"] = twin.best(**reference[0])
            for model, row_name in rows_of.items():
                llm = reference_model(reference) if model == "reference" else \
                    agent.chat_model(model, args.url, max_tokens=args.max_tokens, reasoning=args.reasoning, use_tools=args.tools)
                a = agent.advise(llm, twin, text, turns=args.turns)
                told = dict(said=a["said"], conditions=a["conditions"], replies=a["replies"], error=a["error"],
                            reply=a["reply"].strip()[:400])
                kind = name.split("-")[0]
                if reference == CANNOT:
                    told["right"] = a["said"] == "cannot"
                else:
                    told.update(a["best"] or {}, extra=extra(twin, key, sign, a["best"] or usual, reference))
                    if kind != "vague":
                        told["right"] = a["best"] is not None and same(twin, a["conditions"], reference[0])
                row[row_name] = told
                total.setdefault((row_name, split(name), kind), []).append(dict(told, twin=twin_name, not_told=row.get("not_told")))
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")

for row_name in rows_of.values():
    print(f"\n{row_name}")
    for s in ("dev", "test", "test2"):
        for kind in kinds:
            cells = total.get((row_name, s, kind), [])
            if not cells:
                continue
            line = f"  {s:<5}{kind:<14}"
            if kind == "vague":             # against the twin that is not told: in how many cells better / the same / worse
                d = [c["extra"] - c["not_told"]["extra"] for c in cells]
                line += f"better {sum(x < 0 for x in d)}, same {sum(x == 0 for x in d)}, worse {sum(x > 0 for x in d)} of {len(d)}"
            else:
                line += f"right {sum(c['right'] for c in cells)}/{len(cells)}"
            twins_ = sorted({c["twin"] for c in cells})
            if kind in ("stated", "vague"):
                line += "   extra, told / not told: " + "  ".join(
                    f"{t} {np.mean([c['extra'] for c in cells if c['twin'] == t]):.3g} / "
                    f"{np.mean([c['not_told']['extra'] for c in cells if c['twin'] == t]):.3g}" for t in twins_)
            else:
                line += "   by twin: " + "  ".join(
                    f"{t} {sum(c['right'] for c in cells if c['twin'] == t)}/{sum(c['twin'] == t for c in cells)}" for t in twins_)
            print(line + f"   second replies: {sum(c['replies'] > 1 for c in cells)}")
