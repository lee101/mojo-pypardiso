# coding: utf-8
"""Mojo port of pypardiso: sparse direct solver with the upstream API."""

from .pardiso_wrapper import PyPardisoSolver
from .scipy_aliases import spsolve, factorized
from .scipy_aliases import pypardiso_solver as ps

__version__ = "0.1.0"

__all__ = ['PyPardisoSolver', 'spsolve', 'factorized', 'ps']
