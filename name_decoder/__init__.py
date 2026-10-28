"""Public decoder APIs loaded on demand; static lexicon inference needs no LLM imports."""
from importlib import import_module

_EXPORTS = {'CatalogNames': ('catalog', 'CatalogNames'), 'NameCandidate': ('catalog', 'NameCandidate'), 'NamePrediction': ('catalog', 'NamePrediction'), 'canonical_name': ('catalog', 'canonical_name'), 'CatalogNameSearch': ('catalog_search', 'CatalogNameSearch'), 'ByT5NameDecoder': ('model', 'ByT5NameDecoder'), 'HyperCLOVANameDecoder': ('hyperclova', 'HyperCLOVANameDecoder')}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    result = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = result
    return result


def __dir__():
    return sorted(set(globals()) | set(__all__))
