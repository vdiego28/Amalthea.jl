#!/usr/bin/env python3
"""Verify transported reference artifacts and structured wheel-test results."""
import argparse
import json
from pathlib import Path
import subprocess

from _validation import pytest_result, verify_oracles


def checked_environment(directory, repository, github_env=None):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    environment = verify_oracles(directory, repository, revision)
    if github_env is not None:
        # GITHUB_ENV uses one key=value per line. Validate before opening it.
        if any('\n' in value or '\r' in value for value in environment.values()):
            raise ValueError('oracle paths cannot contain newlines in GITHUB_ENV')
        with Path(github_env).open('a', encoding='utf-8') as stream:
            for key, value in environment.items():
                stream.write(f'{key}={value}\n')
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    oracles = subparsers.add_parser('oracles')
    oracles.add_argument('directory', type=Path)
    oracles.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    oracles.add_argument('--github-env', type=Path)
    results = subparsers.add_parser('pytest')
    results.add_argument('reports', nargs='+', type=Path)
    arguments = parser.parse_args()
    if arguments.command == 'oracles':
        result = checked_environment(arguments.directory.resolve(), arguments.repository.resolve(),
                                     arguments.github_env)
    else:
        result = {str(path): pytest_result(path) for path in arguments.reports}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
