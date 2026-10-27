"""
Progress marker for MaxwellFDFD.

This module provides the ProgMark class that tracks and reports the elapsed
time of operations during FDFD system construction and solving.
"""

import time


class ProgMark:
    """
    Progress marker that times operations.

    ProgMark marks the progress of operations by measuring elapsed time
    between successive :meth:`mark` calls. It is used throughout
    :func:`maxwell_run` and :func:`build_system` to report timing of each
    computational stage.

    Attributes
    ----------
    _t0 : float
        Initial timestamp (from :func:`time.perf_counter`).
    _t_prev : float
        Elapsed time at the previous measurement.

    Examples
    --------
    >>> from maxwellfdfd.io import ProgMark
    >>> pm = ProgMark()
    >>> # ... do some work ...
    >>> pm.mark('grid generation')
    time elapsed: ... sec in total, ... sec for grid generation
    """

    def __init__(self) -> None:
        """Initialize the progress marker and start timing."""
        self._t0: float = time.perf_counter()
        self._t_prev: float = 0.0

    def mark(self, operation_name: str) -> None:
        """
        Mark the completion of an operation and report elapsed time.

        Prints the total elapsed time since the ProgMark was created and
        the time spent on the current operation since the previous mark.

        Parameters
        ----------
        operation_name : str
            Human-readable name of the operation that just completed.
        """
        t_curr: float = time.perf_counter() - self._t0
        print(
            f'time elapsed: {t_curr} sec in total, '
            f'{t_curr - self._t_prev} sec for {operation_name}'
        )
        self._t_prev = t_curr
