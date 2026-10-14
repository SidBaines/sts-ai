import unittest
from types import SimpleNamespace

try:
    from sts_rl.common import FilteringWorker
except ModuleNotFoundError as error:
    if error.name not in {'torch', 'numpy'}:
        raise
    FilteringWorker = None


@unittest.skipIf(FilteringWorker is None, 'Optional RL runtime unavailable')
class WorkerFaultLinkTests(unittest.TestCase):
    def test_wrapper_forwards_current_fault_record(self):
        inner = SimpleNamespace(last_fault_path='first.json')
        wrapper = FilteringWorker(inner)
        self.assertEqual(wrapper.last_fault_path, 'first.json')
        inner.last_fault_path = 'second.json'
        self.assertEqual(wrapper.last_fault_path, 'second.json')
