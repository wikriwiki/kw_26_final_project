"""Request a complete graph backup while all agent workers are quiescent."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

# Start early enough to drain bounded requests and upload before 12 hours.
INTERVAL_SECONDS = 10 * 3600


def due(run_dir, now=None):
    if not os.environ.get('SIM_POST_DAY_BACKUP_HOOK'):
        return False
    receipt = Path(run_dir) / 'recoverable_backup.json'
    if not receipt.is_file():
        return True
    record = json.loads(receipt.read_text(encoding='utf-8'))
    verified = datetime.fromisoformat(record['verified_at_utc'])
    return ((now or datetime.now(timezone.utc)) - verified).total_seconds() >= INTERVAL_SECONDS


def save_if_due(run_dir, day):
    """Caller MUST have drained all workers before this function is called."""
    if due(run_dir):
        hook = Path(os.environ['SIM_POST_DAY_BACKUP_HOOK'])
        if not hook.is_file() or hook.is_symlink():
            raise ValueError('Invalid recoverable checkpoint hook')
        subprocess.run([sys.executable, str(hook), '--progress', str(day), str(Path(run_dir).resolve())],
                       check=True, timeout=90 * 60)
