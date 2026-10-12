"""Remove only the paused original scheduler after its preserved training ends."""
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results/eikonal_five_seed_device3'
old = json.loads((root / 'expansion/previous_status.json').read_text())
names = [name for name, job in old['jobs'].items() if job['state'] == 'training']
assert len(names) == 4

while True:
    ready = []
    for name in names:
        try:
            metadata = json.loads((root / 'checkpoints' / name / 'training.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {}
        ready.append(metadata.get('completed', False))
    if all(ready):
        break
    time.sleep(5)

time.sleep(5)  # let the final worker return and release its CUDA context
container = 'lie-eikonal-five-seed-device3'
result = subprocess.run(['docker', 'kill', '--signal=KILL', container], capture_output=True, text=True)
(root / 'expansion/legacy_cleanup.log').write_text(result.stdout + result.stderr)
print('Preserved training finished; original paused scheduler cleanup exit:', result.returncode, flush=True)
