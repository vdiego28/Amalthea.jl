"""Launch installed examples with OS-enforced network isolation (development only)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from _validation import run, save_json


WINDOWS_FIREWALL = r'''
param([Parameter(Mandatory=$true)][string]$ConfigPath)
$ErrorActionPreference = 'Stop'
$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$names = @()
try {
    if (@(Get-NetFirewallProfile | Where-Object { -not $_.Enabled }).Count -ne 0) {
        throw 'Offline acceptance requires enabled firewall profiles; no policy will be changed.'
    }
    foreach ($program in $config.programs) {
        foreach ($direction in @('Inbound', 'Outbound')) {
            $name = $config.rule_prefix + '-' + $names.Count
            $names += $name
            New-NetFirewallRule -Name $name -DisplayName $name -Program $program `
                -Direction $direction -Action Block -Enabled True -Profile Any | Out-Null
            $rule = Get-NetFirewallRule -PolicyStore ActiveStore -Name $name
            if ($rule.Action -ne 'Block' -or $rule.Enabled -ne 'True') {
                throw "Firewall rule not active: $name"
            }
        }
    }
    $env:PATH = $config.empty_path
    $arguments = @($config.arguments)
    & $config.python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Offline example process failed: $LASTEXITCODE" }
} finally {
    foreach ($name in $names) {
        $rule = Get-NetFirewallRule -Name $name -ErrorAction SilentlyContinue
        if ($null -ne $rule) { $rule | Remove-NetFirewallRule -ErrorAction Stop }
    }
}
'''


def offline(python, examples, output, *, environment, cwd, ephemeral_ci=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    launcher = Path(__file__).with_name('installed_smoke.py').resolve()
    kind = {'linux': 'linux', 'darwin': 'macos', 'win32': 'windows'}.get(sys.platform)
    if kind is None:
        raise RuntimeError(f'unsupported offline platform: {sys.platform}')
    arguments = ['-I', '-X', 'utf8', str(launcher), '--examples', str(examples), '--output', str(output), '--isolation', kind]
    empty_path = os.path.join(os.path.abspath(os.sep), 'nonexistent')
    isolated = environment | {'PATH': empty_path}
    log = output.with_suffix('.log')
    if kind == 'linux':
        unshare = shutil.which('unshare', path=environment.get('PATH'))
        if not unshare:
            raise RuntimeError('unshare is required for offline Linux acceptance')
        # Hosted Ubuntu restricts unprivileged user namespaces. Its ephemeral
        # runners provide sudo; local runs use their own user namespace.
        if ephemeral_ci and environment.get('RUNNER_ENVIRONMENT') == 'github-hosted':
            sudo = shutil.which('sudo', path=environment.get('PATH'))
            env = shutil.which('env', path=environment.get('PATH'))
            if not sudo or not env:
                raise RuntimeError('hosted offline acceptance requires sudo and env')
            command = [sudo, '-n', unshare, '--net', env, f'PATH={empty_path}', str(python), *arguments]
            run(command, cwd=cwd, environment=environment, log=log)
        else:
            command = [unshare, '--user', '--map-root-user', '--net', str(python), *arguments]
            run(command, cwd=cwd, environment=isolated, log=log)
    elif kind == 'macos':
        sandbox = Path('/usr/bin/sandbox-exec')
        if not sandbox.is_file():
            raise RuntimeError('sandbox-exec is required for offline macOS acceptance')
        run([sandbox, '-p', '(version 1) (allow default) (deny network*)', python, *arguments],
            cwd=cwd, environment=isolated, log=log)
    else:
        if not (ephemeral_ci and environment.get('GITHUB_ACTIONS') == 'true'
                and environment.get('RUNNER_ENVIRONMENT') == 'github-hosted'):
            raise RuntimeError('Windows firewall isolation is restricted to explicit ephemeral hosted CI')
        powershell = shutil.which('pwsh', path=environment.get('PATH'))
        if not powershell:
            raise RuntimeError('PowerShell is required for offline Windows acceptance')
        probe = subprocess.check_output([str(python), '-I', '-c',
            'import json,sys; print(json.dumps([sys.executable,sys._base_executable]))'],
            cwd=cwd, env=environment, text=True)
        config = {'python': str(python), 'arguments': arguments, 'empty_path': empty_path,
                  'programs': list(dict.fromkeys(json.loads(probe))),
                  'rule_prefix': 'amalthea-offline-' + uuid.uuid4().hex}
        config_path = output.with_suffix('.firewall.json')
        script_path = output.with_suffix('.ps1')
        save_json(config_path, config)
        script_path.write_text(WINDOWS_FIREWALL, encoding='utf-8')
        run([powershell, '-NoProfile', '-NonInteractive', '-File', script_path, '-ConfigPath', config_path],
            cwd=cwd, environment=environment, log=log)
    result = json.loads(output.read_text(encoding='utf-8'))
    if result.get('status') != 'passed' or result.get('isolation', {}).get('mechanism') != kind:
        raise RuntimeError(f'offline examples did not complete: {output}')
    return result
