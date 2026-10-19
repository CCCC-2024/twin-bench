"""Box controller (runs on the physical side), written from the dataset READMEs' description.

Parameters: temperature_desired, lower_bound, heating_time, heating_gap.
  CoolingDown: heater off; start heating when T <= desired - lower_bound.
  Heating:     heater on; stop when T > desired, or after heating_time seconds (then Waiting).
  Waiting:     heater off for heating_gap seconds; then heat again if T <= desired.
The 2023 experiments used desired 38, lower_bound 2, heating_time 60000 (never times out), gap 5.
"""


class Controller:
    def __init__(self, desired=38.0, lower_bound=2.0, heating_time=60000.0, heating_gap=5.0):
        self.desired, self.lower_bound = desired, lower_bound
        self.heating_time, self.heating_gap = heating_time, heating_gap
        self.state, self.next_time, self.heater_on = "CoolingDown", -1.0, False

    def set(self, desired=None, lower_bound=None):
        if desired is not None:
            self.desired = float(desired)
        if lower_bound is not None:
            self.lower_bound = float(lower_bound)

    def copy(self):
        c = Controller(self.desired, self.lower_bound, self.heating_time, self.heating_gap)
        c.state, c.next_time, c.heater_on = self.state, self.next_time, self.heater_on
        return c

    def step(self, t, T):
        if self.state == "CoolingDown":
            if T <= self.desired - self.lower_bound:
                self.state, self.heater_on, self.next_time = "Heating", True, t + self.heating_time
        elif self.state == "Heating":
            if 0 < self.next_time <= t:
                self.state, self.heater_on, self.next_time = "Waiting", False, t + self.heating_gap
            elif T > self.desired:
                self.state, self.heater_on, self.next_time = "CoolingDown", False, -1.0
        elif self.state == "Waiting":
            if 0 < self.next_time <= t:
                if T <= self.desired:
                    self.state, self.heater_on, self.next_time = "Heating", True, t + self.heating_time
                else:
                    self.state, self.heater_on, self.next_time = "CoolingDown", False, -1.0
        return self.heater_on
