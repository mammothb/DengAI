import argparse
from pathlib import Path

from src.datapipeline.loader import Loader

def run_experiment(params):
    print(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path)

    params = dict()

    args = parser.parse_args()


    print(args)

    run_experiment(
        params,
    )
