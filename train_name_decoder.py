r"""Train ByT5 to predict catalog vessel names (no submission ZIP creation).

PowerShell: .\.venv-server\Scripts\python.exe train_name_decoder.py --help
Implementation and experiment logging live in experiments/train_name_decoder.py.
"""
from experiments.train_name_decoder import main


if __name__ == "__main__":
    main()
