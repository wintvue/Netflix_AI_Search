#!/usr/bin/env python3
"""Typed errors for the search backend.

Kept in their own module so the API layer can map them to status codes without
importing the model or database machinery.
"""


class SearchBackendError(Exception):
    """Base class for failures in a dependency the search pipeline needs."""


class ConfigurationError(SearchBackendError):
    """A required setting is missing or invalid."""
