"""Bundle discovery must not enumerate the QuickDraw dataset."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'kaggle_offline', Path(__file__).resolve().parents[1] / 'test' / 'kaggle_offline.py')
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)


class BundleDiscoveryTests(unittest.TestCase):
    def test_nested_bundle_found_without_listing_dataset_or_bundle_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'datasets' / 'owner' / 'quickdraw' / 'QuickDraw'
            (dataset / 'sketch' / 'alarm_clock').mkdir(parents=True)
            (dataset / 'photo' / 'alarm_clock').mkdir(parents=True)
            output = root / 'kernels' / 'owner' / 'notebook' / 'seco_quickdraw_timestamp'
            manifest = output / 'offline_bundle' / 'bundle_manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}')
            visited = []
            scandir = offline.os.scandir

            def checked_scandir(path):
                path = Path(path)
                self.assertFalse(path.is_relative_to(dataset), f'Entered dataset: {path}')
                self.assertFalse(path.is_relative_to(output), f'Entered located output: {path}')
                visited.append(path)
                return scandir(path)

            with patch.object(offline.os, 'scandir', side_effect=checked_scandir), redirect_stdout(io.StringIO()):
                found = offline.find_bundle_manifests(root)
            self.assertEqual(found, [manifest])
            self.assertLess(len(visited), 12)

    def test_separate_attachments_are_reported_for_ambiguity_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = []
            for name in ('first_output', 'second_output'):
                manifest = root / name / 'offline_bundle' / 'bundle_manifest.json'
                manifest.parent.mkdir(parents=True)
                manifest.write_text('{}')
                manifests.append(manifest)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(offline.find_bundle_manifests(root), manifests)

    def test_unpaired_image_directories_are_pruned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('sketch', 'photo', 'source', 'wheels'):
                (root / 'dataset' / name / 'deep').mkdir(parents=True)
            scandir = offline.os.scandir
            visited = []

            def checked_scandir(path):
                visited.append(Path(path))
                return scandir(path)

            with patch.object(offline.os, 'scandir', side_effect=checked_scandir), redirect_stdout(io.StringIO()):
                self.assertEqual(offline.find_bundle_manifests(root), [])
            self.assertEqual(visited, [root])


if __name__ == '__main__':
    unittest.main()
