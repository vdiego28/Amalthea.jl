#!/usr/bin/env python3
"""Export a complete, hashed Julia reference artifact for wheel-test consumers."""
import argparse
import subprocess
from pathlib import Path
import tomllib

from _validation import (FIXTURES, cpu_environment, digest, reference_sources,
                         run, save_json, utcnow, verify_oracles)


def export(repository, output, julia):
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    state = {'format_version': 1, 'status': 'running', 'revision': revision,
             'started': utcnow(), 'sources': reference_sources(repository), 'families': []}
    environment = cpu_environment()
    # Developer fixture selectors must not turn a partial export into a CI artifact.
    for key in list(environment):
        if key.startswith(('AMALTHEA_MODAL_', 'AMALTHEA_CALLBACK_', 'AMALTHEA_MIXTURE_')):
            del environment[key]
        elif key.startswith('AMALTHEA_USE_RUST_'):
            environment[key] = '0'
    for key in ('NATIVE', 'STEPPER', 'DISPERSION', 'IONISATION', 'RAMAN', 'QDHT'):
        environment[f'AMALTHEA_USE_RUST_{key}'] = '0'
    save_json(output/'export-state.json', state)
    try:
        run([julia, '--startup-file=no', '--version'], cwd=repository,
            environment=environment, log=output/'logs/julia-version.log')
        for key, (directory, exporters) in FIXTURES.items():
            for name in exporters:
                script = repository/'python-native/tools'/f'export_{name}_oracle.jl'
                run([julia, '--startup-file=no', f'--project={repository}', script, output/directory],
                    cwd=repository, environment=environment, log=output/'logs'/f'{name}.log')
            if not any(path.is_file() for path in (output/directory).rglob('*')):
                raise RuntimeError(f'exporter produced no fixture files: {key}')
            state['families'].append(key)
            save_json(output/'export-state.json', state)
        state['files'] = {path.relative_to(output).as_posix(): digest(path)
                          for directory, _ in FIXTURES.values()
                          for path in sorted((output/directory).rglob('*')) if path.is_file()}
        state['julia'] = (output/'logs/julia-version.log').read_text().strip()
        state['coolprop'] = tomllib.loads((output/'materials/metadata.toml').read_text())['coolprop']
        if state['coolprop'] != '7.2.0':
            raise RuntimeError(f"oracle CoolProp must be 7.2.0, got {state['coolprop']}")
        if state['sources'] != reference_sources(repository):
            raise RuntimeError('reference source changed during export')
        state.update(status='complete', finished=utcnow())
        save_json(output/'manifest.json', state)
        verify_oracles(output, repository, revision)
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished=utcnow())
        # Consumers require the completion manifest; a failed export cannot leave one.
        (output/'manifest.json').unlink(missing_ok=True)
        raise
    finally:
        save_json(output/'export-state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True, help='New, not already existing directory')
    parser.add_argument('--julia', default='julia')
    arguments = parser.parse_args()
    export(arguments.repository.resolve(), arguments.output.resolve(), arguments.julia)


if __name__ == '__main__':
    main()
