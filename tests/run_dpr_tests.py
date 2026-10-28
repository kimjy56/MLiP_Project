"""Run DPR function-style and unittest tests with the standard library.

Useful in the pinned server environment, where pytest is not a dependency.
The same test modules remain discoverable by pytest where it is installed.
"""
from pathlib import Path
import importlib
import inspect
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    suite = unittest.TestSuite()
    names = sys.argv[1:] or ['tests.test_dpr03', 'tests.test_phonetic_encoder',
                             'tests.test_dpr_hybrid', 'tests.test_dense_features']
    for name in names:
        module = importlib.import_module(name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for test_name, fn in inspect.getmembers(module, inspect.isfunction):
            if test_name.startswith('test_') and fn.__module__ == module.__name__:
                suite.addTest(unittest.FunctionTestCase(fn, description=f'{name}.{test_name}'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
