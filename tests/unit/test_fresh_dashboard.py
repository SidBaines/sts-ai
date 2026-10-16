import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fresh_dashboard',ROOT/'scripts/rl/fresh_dashboard.py')
dashboard=importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class FreshDashboardTests(unittest.TestCase):
    def test_state_limits_paths_to_declared_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'
            root.mkdir()
            outside=Path(tmp)/'secret.json'
            outside.write_text('{"secret":"must not appear"}')
            (root/'config.json').write_text('{"phase":"combat"}')
            (root/'summary.json').symlink_to(outside)
            (root/'status.json').write_text('{"status":"supervised_training"}')
            (root/'metrics.jsonl').write_text('{"step":1}\npartial')
            state=dashboard.study_state(root)
            self.assertNotIn('must not appear',json.dumps(state))
            self.assertEqual(len(state['runs']),1)
            self.assertEqual(state['runs'][0]['metrics'],[{'step':1}])

    def test_localhost_get_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'status.json').write_text('{"status":"completed"}')
            try:
                server=dashboard.make_server(tmp)
            except PermissionError as exc:
                self.skipTest('Sandbox denied localhost bind; orchestrator host test required: '+str(exc))
            self.assertEqual(server.server_address[0],'127.0.0.1')
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            url='http://127.0.0.1:'+str(server.server_port)
            try:
                with urlopen(url+'/api/state',timeout=2) as response:
                    self.assertTrue(json.load(response)['read_only'])
                with urlopen(url,timeout=2) as response:
                    self.assertIn(b'Supervised validation',response.read())
                for path,method in (('/api/state','POST'),('/../secret','GET'),('/actions','GET')):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(Request(url+path,data=b'{}' if method=='POST' else None,method=method),timeout=2)
                    self.assertIn(error.exception.code,(404,501))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

if __name__=='__main__':
    unittest.main()
