# Update Benchmark Task

**Trigger:** When PyPI releases a new version of pymc-marketing, meridian, or google-meridian

## What this task does

Checks if any benchmarked tool has released a new version since the last run.
If so, re-runs the benchmark and opens a PR updating the README leaderboard.

## Step 0: Load credentials

```python
import os
with open('C:/Users/iamni/OneDrive/Documents/quokka/.env') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
token = os.environ['QUOKKA_GITHUB_TOKEN']
```

## Step 1: Check for new versions on PyPI

```python
import urllib.request, json

packages = ['pymc-marketing', 'google-meridian']
latest_versions = {}

for pkg in packages:
    url = f'https://pypi.org/pypi/{pkg}/json'
    try:
        resp = urllib.request.urlopen(url)
        data = json.loads(resp.read())
        latest_versions[pkg] = data['info']['version']
        print(f'{pkg}: {latest_versions[pkg]}')
    except Exception as e:
        print(f'Could not fetch {pkg}: {e}')
```

## Step 2: Compare against last run versions

Read `results/` directory for the most recent summary.json and compare versions.
If any tool has a newer version than the last run, proceed. Otherwise exit.

```python
import os
from pathlib import Path

results_dir = Path('C:/Users/iamni/OneDrive/Documents/mmm-bench/results')
run_dirs = sorted(results_dir.glob('*/summary.json'), reverse=True)

if run_dirs:
    with open(run_dirs[0]) as f:
        last_run = json.load(f)
    last_versions = {r['tool']: r['version'] for r in last_run['results']}
    print('Last run versions:', last_versions)
else:
    print('No previous runs found - will run benchmark fresh')
    last_versions = {}
```

## Step 3: Run the benchmark

```python
import subprocess

result = subprocess.run(
    ['python', 'benchmark.py', '--update-readme'],
    cwd='C:/Users/iamni/OneDrive/Documents/mmm-bench',
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print('BENCHMARK FAILED:', result.stderr)
    exit(1)
```

## Step 4: Commit and push a branch

```python
import subprocess, datetime

repo_dir = 'C:/Users/iamni/OneDrive/Documents/mmm-bench'
branch = f"quokka/benchmark-{datetime.date.today().isoformat()}"

subprocess.run(['git', '-C', repo_dir, 'checkout', '-b', branch], check=True)
subprocess.run(['git', '-C', repo_dir, 'add', 'README.md', 'results/'], check=True)
subprocess.run([
    'git', '-C', repo_dir, 'commit', '-m',
    f'benchmark: update leaderboard {datetime.date.today()}\n\nCo-Authored-By: Quokka <quokka@simba-mmm.com>'
], check=True)

# Push using Quokka token
remote = f'https://simba-quokka:{token}@github.com/simba-quokka/mmm-bench.git'
subprocess.run(['git', '-C', repo_dir, 'push', '-u', remote, branch], check=True)
```

## Step 5: Open a PR

```python
import json, urllib.request

# Build PR body summarising what changed
changed_versions = {
    pkg: f"{last_versions.get(pkg, 'N/A')} → {ver}"
    for pkg, ver in latest_versions.items()
    if last_versions.get(pkg) != ver
}
version_summary = '\n'.join(f'- {pkg}: {change}' for pkg, change in changed_versions.items())

pr_body = f"""## Benchmark update

**Updated tool versions:**
{version_summary}

**Scenarios run:** simple, complex, data-scarce, adversarial

See `results/` directory for full JSON output.

---
*Automated by [Quokka](https://github.com/simba-quokka)*
"""

query = '''mutation CreatePR($repoId: ID!, $base: String!, $head: String!, $title: String!, $body: String!) {
  createPullRequest(input: {
    repositoryId: $repoId, baseRefName: $base, headRefName: $head, title: $title, body: $body
  }) {
    pullRequest { url number }
  }
}'''

payload = {
    'query': query,
    'variables': {
        'repoId': 'MMMBENCH_REPO_ID',  # fill after repo creation
        'base': 'main',
        'head': branch,
        'title': f'benchmark: leaderboard update {datetime.date.today()}',
        'body': pr_body
    }
}

data = json.dumps(payload).encode()
req = urllib.request.Request('https://api.github.com/graphql', data=data, headers={
    'Authorization': f'bearer {token}',
    'Content-Type': 'application/json',
})
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
pr_url = result['data']['createPullRequest']['pullRequest']['url']
print(f'PR created: {pr_url}')
```

## Step 6: Post to simba-mmm Discussions

Post a summary of the benchmark update to the MMM community:

Use the daily-content task pattern but specifically about the benchmark results.
Highlight if any tool improved or regressed, and link to the PR.

Sign off: *- Quokka*
