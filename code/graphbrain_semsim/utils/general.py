import logging
from itertools import groupby


logger = logging.getLogger(__name__)


def frange(start, stop, step, include_stop: bool = True) -> list[float]:
    factor: float = 1.0 / step
    try:
        factor = int(factor)
    except ValueError:
        raise ValueError("Step must be of the form that 1.0 / step is an integer")

    frange_inlc_stop: list[float] = [
        x / factor for x in (list(range(int(start * factor), int(stop * factor))) + [stop * factor])
    ]

    if include_stop:
        return frange_inlc_stop
        
    return frange_inlc_stop[:-1]


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


