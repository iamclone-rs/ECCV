import argparse
from pathlib import Path
import tempfile
import unittest
import json
import runpy
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.data_config import UNSEEN_CLASSES
from src.dataset_protocol import category_split, image_paths
from src.losses import loss_fn
from src.retrieval import retrieval_metrics
from src.sketchy_dataset import TrainDataset, ValidDataset
from PIL import Image


class QuickDrawTests(unittest.TestCase):
    def test_image_index_filters_files_without_path_is_file_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'one.PNG'
            image.write_bytes(b'image')
            (root / 'README.txt').write_text('metadata')
            (root / 'folder.png').mkdir()
            with patch.object(Path, 'is_file', side_effect=AssertionError('Per-image Path.stat call')):
                self.assertEqual(image_paths(root), [str(image)])

    def test_training_and_validation_reuse_cached_classnames(self):
        from src.model import ZS_SBIR
        from unittest.mock import Mock
        model = ZS_SBIR.__new__(ZS_SBIR)
        torch.nn.Module.__init__(model)
        model.args = argparse.Namespace(visualize=False)
        model.classname = ['seen_one', 'seen_two']
        model.log = Mock()
        model.forward = Mock(return_value='features')
        model.model = torch.nn.Module()
        model.model.extract_feature = Mock(return_value=torch.ones(2, 5))
        model.val_step_outputs_sk = []
        model.val_step_outputs_ph = []
        batch = (torch.ones(2, 3, 4, 4), torch.tensor([0, 1]))
        with patch('src.model.get_all_categories', side_effect=AssertionError('Repeated filesystem scan')), \
                patch('src.model.loss_fn', return_value=torch.tensor(1.)):
            model.training_step(batch, 0)
            model.validation_step(batch, 0, 0)
            model.validation_step(batch, 0, 1)
        model.forward.assert_called_once_with(batch, model.classname)
        self.assertEqual(len(model.val_step_outputs_sk), 1)
        self.assertEqual(len(model.val_step_outputs_ph), 1)
        for call in model.model.extract_feature.call_args_list:
            self.assertEqual(call.kwargs['classname'], model.classname)

    def test_backbone_freezing_includes_attention_and_position_embeddings(self):
        from src.model import freeze_all_but_bn
        backbone = torch.nn.Module()
        backbone.attention = torch.nn.MultiheadAttention(8, 2)
        backbone.norm = torch.nn.LayerNorm(8)
        backbone.positional_embedding = torch.nn.Parameter(torch.randn(5, 8))
        backbone.apply(freeze_all_but_bn)
        for name, parameter in backbone.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith('norm.'))

    def make_data(self, root):
        # The user-supplied 30-class split plus 80 disjoint seen categories.
        categories = list(UNSEEN_CLASSES['quickdraw']) + [f'seen_{i:02d}' for i in range(80)]
        for modality in ('sketch', 'photo'):
            for category in categories:
                directory = Path(root) / modality / category
                directory.mkdir(parents=True)
                Image.new('RGB', (12, 12), 'white').save(directory / 'one.png')
        return argparse.Namespace(root=root, dataset='quickdraw', use_classes=80,
            seed=42, max_size=224, visualize=False, gzs=False)

    def test_user_split_and_dataset_labels(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.make_data(root)
            # Ignore notebook metadata and unrelated files.
            (Path(root) / 'sketch' / '.ipynb_checkpoints').mkdir()
            (Path(root) / 'sketch' / 'README.txt').write_text('metadata')
            seen, unseen = category_split(args)
            self.assertEqual(unseen, UNSEEN_CLASSES['quickdraw'])
            self.assertEqual(len(seen), 80)
            self.assertFalse(set(seen) & set(unseen))
            train = TrainDataset(args)
            query = ValidDataset(args, mode='sketch')
            gallery = ValidDataset(args)
            self.assertEqual(train.all_categories, seen)
            self.assertEqual(len(train), 80)
            self.assertEqual(train[0][-1], 0)
            self.assertEqual(query[0][-1], gallery[0][-1])
            self.assertEqual(query.unseen_classes, unseen)

    def test_missing_test_class_fails(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.make_data(root)
            path = Path(root) / 'sketch' / 'alarm_clock'
            (path / 'one.png').unlink()
            path.rmdir()
            with self.assertRaisesRegex(ValueError, 'missing'):
                category_split(args)

    def test_kaggle_preflight_accepts_exact_user_split_without_weights(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_data(root)
            output = str(Path(root) / 'output')
            with patch('sys.argv', ['kaggle_train.py', '--root', root,
                                    '--output_dir', output, '--check_only']):
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'test' / 'kaggle_train.py'), run_name='__main__')
            report = json.loads((Path(output) / 'quickdraw_preflight.json').read_text())
            self.assertEqual(report['test_classes'], UNSEEN_CLASSES['quickdraw'])
            self.assertEqual(len(report['train_classes']), 80)

    def test_folder_separator_alias_does_not_change_membership(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.make_data(root)
            for modality in ('photo', 'sketch'):
                parent = Path(root) / modality
                (parent / 'alarm_clock').rename(parent / 'alarm clock')
            seen, unseen = category_split(args)
            self.assertIn('alarm clock', unseen)
            self.assertNotIn('alarm clock', seen)

    def test_retrieval_has_exact_ap_and_precision_denominator(self):
        query = torch.tensor([[1., 0.], [0., 1.]])
        gallery = torch.tensor([[1., 0.], [0.8, 0.6], [0., 1.], [-1., 0.]])
        # Query 0 relevant ranks 1,3 -> AP=5/6; query 1 has no positives.
        ap, precision = retrieval_metrics(query, gallery, [0, 9], [0, 1, 0, 2],
                                          precision_k=4, query_batch_size=1)
        self.assertAlmostEqual(ap.item(), 5 / 12, places=6)
        self.assertAlmostEqual(precision.item(), 1 / 4, places=6)
        # P@k divides by k even when only one relevant result exists.
        _, precision = retrieval_metrics(query[:1], gallery, [1], [0, 1, 0, 2], precision_k=4)
        self.assertAlmostEqual(precision.item(), 1 / 4, places=6)

    def test_paper_loss_matches_equations_and_frozen_targets(self):
        torch.manual_seed(1)
        photo = torch.randn(3, 5, requires_grad=True)
        sketch = torch.randn(3, 5, requires_grad=True)
        frozen_photo = torch.randn(3, 5, requires_grad=True)
        frozen_sketch = torch.randn(3, 5, requires_grad=True)
        negative = F.normalize(torch.randn(3, 5), dim=-1)
        labels = torch.arange(3)
        photo_logits, sketch_logits = torch.randn(3, 3), torch.randn(3, 3)
        args = argparse.Namespace(loss_protocol='paper', temperature=.07, triplet_margin=.3,
                                  lambda_class=1., lambda_consistency=1.)
        pn, sn = F.normalize(photo, dim=-1), F.normalize(sketch, dim=-1)
        features = (pn, sn, frozen_photo, frozen_sketch, negative, labels,
                    photo_logits, sketch_logits, photo, sketch)
        actual = loss_fn(args, None, features)
        similarity = sn @ pn.T / .07
        expected = F.cross_entropy(photo_logits, labels) + F.cross_entropy(sketch_logits, labels)
        expected += torch.relu(.3 + (1 - (sn * pn).sum(-1)) - (1 - (sn * negative).sum(-1))).mean()
        expected += .5 * (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels))
        for clean, reference in ((pn, frozen_photo), (sn, frozen_sketch)):
            expected += F.cross_entropy(F.normalize(reference.detach(), dim=-1) @ clean.T / .07, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(frozen_photo.grad)
        self.assertIsNone(frozen_sketch.grad)
        self.assertTrue(torch.isfinite(photo.grad).all())


if __name__ == '__main__':
    unittest.main()
