"""CPU-built bundles must reuse the GPU notebook's installed CUDA stack."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'kaggle_offline_versions', Path(__file__).resolve().parents[1] / 'test' / 'kaggle_offline.py')
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


class TorchVersionTests(unittest.TestCase):
    def test_cpu_bundle_produces_cuda_runtime_constraints(self):
        expected = {'torch': '2.11.0+cpu', 'torchvision': '0.26.0+cpu'}
        installed = {'torch': '2.11.0+cu128', 'torchvision': '0.26.0+cu128'}
        with tempfile.TemporaryDirectory() as temporary:
            with patch('importlib.metadata.version', side_effect=installed.__getitem__), redirect_stdout(io.StringIO()):
                constraints = offline.runtime_torch_constraints(expected, temporary)
            self.assertEqual(constraints.read_text(), 'torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n')

    def test_different_release_still_rejected(self):
        for actual in ('2.10.0+cu128', '2.11.0.dev20260101+cu128'):
            with self.subTest(actual=actual), tempfile.TemporaryDirectory() as temporary:
                with patch('importlib.metadata.version', return_value=actual), redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'release differs'):
                        offline.runtime_torch_constraints({'torch': '2.11.0+cpu'}, temporary)
                self.assertFalse((Path(temporary) / 'eccv_runtime_torch_constraints.txt').exists())

    def test_offline_install_does_not_use_builder_cpu_constraints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'offline_bundle'
            source = bundle / 'source' / 'ECCV' / 'src' / 'main_train.py'
            weights = bundle / 'clip_cache' / 'ViT-B-32.pt'
            for path, contents in ((source, b'# fixture source'), (weights, b'fixture weights')):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)
            (bundle / 'torch_constraints.txt').write_text('torch==2.11.0+cpu\n')
            manifest = {'task': 'seco_sbir_quickdraw', 'schema_version': 1,
                        'python_major_minor': list(sys.version_info[:2]),
                        'torch_stack': {'torch': '2.11.0+cpu'},
                        'runtime_requirements': ['pytorch-lightning'], 'files': []}
            for path in (source, weights):
                manifest['files'].append({'path': path.relative_to(bundle).as_posix(),
                                          'size': path.stat().st_size,
                                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            (bundle / 'bundle_manifest.json').write_text(json.dumps(manifest))
            working = root / 'working'
            with patch.object(offline, 'WORKING', working), patch.object(offline, 'PROJECT', working / 'ECCV'), \
                    patch.dict('os.environ', {'ECCV_BUNDLE': str(bundle)}), \
                    patch('importlib.metadata.version', return_value='2.11.0+cu128'), \
                    patch.object(offline.subprocess, 'run') as command, redirect_stdout(io.StringIO()):
                offline.main()
            pip_command = command.call_args_list[0].args[0]
            self.assertNotIn(str(bundle / 'torch_constraints.txt'), pip_command)
            constraints = working / 'eccv_runtime_torch_constraints.txt'
            self.assertIn(str(constraints), pip_command)
            self.assertEqual(constraints.read_text(), 'torch==2.11.0+cu128\n')
            self.assertTrue((working / 'ECCV' / 'src' / 'main_train.py').is_file())


if __name__ == '__main__':
    unittest.main()
