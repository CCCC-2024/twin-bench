"""The two interfaces a system implements to run on dtcore."""


class Source:
    """Physical side: a real sensor feed, a replay of recorded data, or a simulator."""

    kind = "replay"          # "replay", "sim" or "sensor"; a replay cannot be changed by actions

    def observe(self):
        """Return the next observation as a dict with at least 't', or None when there is no more data."""
        raise NotImplementedError

    def act(self, action):
        """Receive a decision from the twin (digital -> physical). A replay can only record it."""

    def advance(self):
        """Move to the next sample (a simulator integrates its dynamics here)."""


class Twin:
    """Digital side. Declare in `capabilities` which of update / predict / whatif / decide it supports."""

    capabilities = ("update",)

    def update(self, obs):
        """Use one observation to update the digital state; return the estimate as a dict."""
        raise NotImplementedError

    def predict(self, horizon, **scenario):
        """Forecast the quantities of interest; a non-empty scenario makes it a what-if."""
        raise NotImplementedError

    def decide(self):
        """Return an action for the physical side, or None (no automatic decision)."""
        return None
