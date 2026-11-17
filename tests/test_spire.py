import os
import tempfile
import unittest

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tf_keras as keras

from spire import spire
from spire.callbacks import create_callbacks_from_model


def changed_after(frames, split, seed):
    rng = np.random.default_rng(seed)
    later = frames.copy()
    later[:, split:] = rng.normal(size=later[:, split:].shape).astype(np.float32)
    return later


class Conv3DTest(unittest.TestCase):

    def model(self, causal):
        x = keras.Input((6, 8, 8, 2))
        return keras.Model(x, spire.conv3d(x, 4, (3, 3, 3), (1, 1, 1), "conv", causal))

    def test_causal_output_ignores_later_frames(self):
        model = self.model(causal=True)
        frames = np.random.default_rng(0).normal(size=(1, 6, 8, 8, 2)).astype(np.float32)
        a = model(frames).numpy()
        b = model(changed_after(frames, 3, 1)).numpy()
        np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-6)

    def test_centered_output_uses_later_frames(self):
        model = self.model(causal=False)
        frames = np.random.default_rng(0).normal(size=(1, 6, 8, 8, 2)).astype(np.float32)
        a = model(frames).numpy()
        b = model(changed_after(frames, 3, 1)).numpy()
        self.assertGreater(float(np.abs(a[:, 2] - b[:, 2]).max()), 1e-3)


class UnitTest(unittest.TestCase):

    def model(self, causal):
        x = keras.Input((6, 8, 8, 8))
        model = keras.Model(x, spire.unit(x, 16, 24, (3, 3, 3), (1, 2, 2), causal, "unit"))
        model.get_layer("unit_gain").set_weights([np.array(1.0, np.float32)])
        return model

    def test_output_shape(self):
        self.assertEqual(self.model(causal=True).output_shape, (None, 6, 4, 4, 16))

    def test_causal_output_ignores_later_frames(self):
        model = self.model(causal=True)
        frames = np.random.default_rng(2).normal(size=(1, 6, 8, 8, 8)).astype(np.float32)
        a = model(frames, training=False).numpy()
        b = model(changed_after(frames, 3, 3), training=False).numpy()
        np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-5)

    def test_squeeze_channels_are_multiples_of_eight(self):
        self.assertEqual([spire.squeeze_channels(n) for n in (24, 80, 112, 184)], [8, 24, 32, 48])


class StemTest(unittest.TestCase):

    def test_stem_halves_the_grid(self):
        x = keras.Input((4, 16, 16, 3))
        self.assertEqual(keras.Model(x, spire.stem(x, True)).output_shape, (None, 4, 8, 8, 8))


class DecoderTest(unittest.TestCase):

    def test_output_is_zero_outside_the_chamber(self):
        mask = np.zeros((16, 16), np.float32)
        mask[4:12, 2:14] = 1.0
        features = keras.Input((2, 1, 1, 8))
        field = keras.Input((2, 16, 16, 1))
        out = spire.decoder(features, field, spire.weight_tensor(mask), 16, 16)
        model = keras.Model([features, field], out)
        rng = np.random.default_rng(4)
        y = model([rng.normal(size=(1, 2, 1, 1, 8)).astype(np.float32),
                   rng.normal(size=(1, 2, 16, 16, 1)).astype(np.float32)]).numpy()
        self.assertEqual(y.shape, (1, 2, 16, 16, 1))
        self.assertEqual(float(np.abs(y[0, :, mask == 0]).max()), 0.0)
        self.assertGreater(float(np.abs(y[0, :, mask == 1]).max()), 0.0)


class LossTest(unittest.TestCase):

    def setUp(self):
        self.weight = np.zeros((4, 5), np.float32)
        self.weight[1:3, 1:4] = 1.0
        rng = np.random.default_rng(5)
        self.y_true = rng.normal(size=(2, 3, 4, 5, 1)).astype(np.float32)
        self.y_pred = rng.normal(size=(2, 3, 4, 5, 1)).astype(np.float32)
        inside = self.weight[None, None, :, :, None] > 0
        self.mse = float(np.mean((self.y_pred - self.y_true)[np.broadcast_to(inside, self.y_true.shape)] ** 2))

    def test_masked_mse_averages_over_the_cells_inside(self):
        loss = spire.masked_mse(self.weight)(tf.constant(self.y_true), tf.constant(self.y_pred))
        self.assertAlmostEqual(float(loss), self.mse, places=5)

    def test_masked_mse_ignores_the_cells_outside(self):
        moved = self.y_pred.copy()
        moved[:, :, 0] += 10.0
        loss = spire.masked_mse(self.weight)(tf.constant(self.y_true), tf.constant(moved))
        self.assertAlmostEqual(float(loss), self.mse, places=5)

    def test_masked_rmse_is_the_root_of_the_masked_mse(self):
        metric = spire.MaskedRMSE(self.weight)
        metric.update_state(tf.constant(self.y_true), tf.constant(self.y_pred))
        self.assertAlmostEqual(float(metric.result()), self.mse ** 0.5, places=5)
        metric.reset_state()
        self.assertEqual(float(metric.result()), 0.0)


class CallbacksTest(unittest.TestCase):

    def test_run_directory_and_three_callbacks(self):
        x = keras.Input((2,))
        model = keras.Model(x, keras.layers.Dense(1)(x), name="spire_reconstruction")
        with tempfile.TemporaryDirectory() as parent:
            callbacks, run_name = create_callbacks_from_model(model, parent_folder=parent)
            self.assertTrue(os.path.isdir(os.path.join(parent, run_name)))
        self.assertTrue(run_name.startswith("spire_reconstruction_"))
        self.assertEqual([type(c).__name__ for c in callbacks],
                         ["ModelCheckpoint", "ReduceLROnPlateau", "EarlyStopping"])


if __name__ == "__main__":
    unittest.main()
