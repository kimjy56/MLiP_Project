"""Check the declared server packages without creating submission archives."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

EXPECTED = {
    'torch': '2.7.1+cu128', 'pandas': '2.0.3', 'numpy': '1.26.4',
    'scipy': '1.15.3', 'scikit-learn': '1.5.2', 'joblib': '1.5.2',
    'threadpoolctl': '3.7.0', 'lightgbm': '4.6.0', 'catboost': '1.2.8',
    'rapidfuzz': '3.14.3', 'Unidecode': '1.4.0', 'transformers': '4.57.6',
    'accelerate': '1.9.0', 'safetensors': '0.6.2', 'sentencepiece': '0.2.0',
    'tqdm': '4.66.5',
}


def os_description():
    """freedesktop_os_release is unavailable on Windows/macOS."""
    if platform.system() == 'Linux':
        try:
            return platform.freedesktop_os_release().get('PRETTY_NAME', platform.platform())
        except OSError:
            pass
    return platform.platform()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    packages = {}
    for name, expected in EXPECTED.items():
        result = {'expected': expected}
        try:
            result['installed'] = importlib.metadata.version(name)
            importlib.import_module({'scikit-learn': 'sklearn', 'Unidecode': 'unidecode'}.get(name, name))
            result['import_ok'] = True
            result['passed'] = result['installed'] == expected
        except Exception as exc:
            result.update(passed=False, error=f'{type(exc).__name__}: {exc}')
        packages[name] = result
    import torch
    import numpy as np
    cpu = torch.from_numpy(np.array([1., 2.], dtype=np.float32))
    assert torch.equal(cpu * 2, torch.tensor([2., 4.]))
    gpu_available = torch.cuda.is_available()
    gpu_test = None
    if gpu_available:
        gpu_test = bool(torch.equal((cpu.cuda() * 2).cpu(), cpu * 2))
    report = {
        'recorded_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'python_expected': '3.11.17', 'python_actual': platform.python_version(),
        'executable': sys.executable, 'platform': platform.platform(),
        'os': os_description(),
        'packages': packages, 'torch_cuda_build': torch.version.cuda,
        'cuda_available': gpu_available, 'cpu_tensor_test': True, 'gpu_tensor_test': gpu_test,
        'passed': platform.python_version() == '3.11.17' and all(v['passed'] for v in packages.values()),
        'scope': 'Python and listed package versions/imports, CPU tensor operation; not an Ubuntu/L4 hardware replica',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
