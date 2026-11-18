"""Small spawn targets for testing native-worker failure handling without loading PyTorch."""
import os
import time

from sts_rl.batched_runner import fault_record


def _crash_fixture(connection, seeds, checkpoint, combat, ooc, games_per_process, reference_records=None, overlays=None):
    for seed in seeds:
        connection.send(("start", seed))
        if seed == 10:
            os._exit(7)
        if seed == 11:
            # Start a second concurrent world, then stop making decisions.
            connection.send(("start", seeds[1]))
            time.sleep(30)
        connection.send(("record", fault_record(seed, "fixture simulator fault")))
    connection.send(("done", None))
    connection.close()


def _fault_fixture(connection, seeds, checkpoint, combat, ooc, games_per_process, reference_records=None, overlays=None):
    connection.send(("start", seeds[0]))
    connection.send(("record", fault_record(seeds[0], "fixture reset fault")))
    connection.close()  # Retire after the fault; untouched worlds need a new process.
