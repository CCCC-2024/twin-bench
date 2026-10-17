"""The loop shared by every system: observe -> update -> decide -> act -> record -> advance."""


def run(source, twin, max_steps=None, recorder=None, before_step=None):
    """Run until the source has no more data or max_steps is reached; return the number of steps.

    before_step(k, source, twin) lets an experiment script change the world at step k
    (open a lid, send a set point) without touching the loop.
    """
    k = 0
    while max_steps is None or k < max_steps:
        if before_step is not None:
            before_step(k, source, twin)
        obs = source.observe()
        if obs is None:
            break
        est = twin.update(obs)
        action = twin.decide() if "decide" in twin.capabilities else None
        if action is not None:
            source.act(action)
        if recorder is not None:
            recorder.add(k, obs, est, action)
        source.advance()
        k += 1
    return k
