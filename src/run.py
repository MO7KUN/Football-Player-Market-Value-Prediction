"""Reproduce the full study from the repository root."""
import argparse
from .collect import download
from .features import build
from .train import train

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-download',action='store_true')
    args=parser.parse_args()
    if not args.skip_download:
        download()
    build()
    train()
    from .report import report
    report()
