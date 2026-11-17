from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
import tf_keras as keras

LN10 = math.log(10.0)


class PhysicsGuidedUpsampling(keras.layers.Layer):

    def __init__(self, coefficient_file, **kwargs):
        super().__init__(**kwargs)
        with np.load(coefficient_file) as d:
            c = {k: d[k] for k in d.files}
        self.grid_shape = tuple(int(n) for n in c.pop("grid_shape"))
        self.n_frames = int(c["cell_a"].shape[0])
        self.probe_cells = c.pop("probe_cells").astype(np.int32)
        gap_cell = c.pop("gap_cell")
        self.gap_cells = np.where(c["low"].ravel())[0].astype(np.int32)
        self.gap_valid = gap_cell.ravel()[self.gap_cells].astype(np.int32)
        self.gap_column = (self.gap_cells // c["low"].shape[1]).astype(np.int32)
        edge = c["table_edge"][..., None]
        c["table_zeta"] = np.where(c["table_z"] < edge, c["table_z"] / edge,
                                   np.where(c["table_z"] < c["z_max"],
                                            1.0 + (c["table_z"] - edge) / (c["z_max"] - edge),
                                            2.0 + (c["table_z"] - c["z_max"])))
        c["cell_bounded"] = c["cell_coef"][2] >= 0
        self.c = {k: tf.constant(v) for k, v in c.items()}

    def operating_condition(self, probes):
        c = self.c
        z = tf.gather_nd(tf.transpose(tf.cast(probes[..., 0], tf.float64), [2, 3, 0, 1]), self.probe_cells)
        m = tf.reduce_mean(z * c["probe_sigma"] + c["probe_mu"], axis=[0, 2])
        knots = c["probe_knots"]
        k = tf.clip_by_value(tf.searchsorted(knots[:, 0], m, side="right") - 1, 0, knots.shape[0] - 2)
        m0, m1 = tf.gather(knots[:, 0], k), tf.gather(knots[:, 0], k + 1)
        l0, l1 = tf.gather(knots[:, 1], k), tf.gather(knots[:, 1], k + 1)
        return tf.exp(l0 + (m - m0) * (l1 - l0) / (m1 - m0))

    def physics_guided_field(self, p):
        c = self.c
        lp = tf.math.log(p) / LN10
        a, b, cc = c["cell_coef"][0], c["cell_coef"][1], c["cell_coef"][2]
        field = tf.where(c["cell_bounded"], a + b * lp - cc * tf.math.log(1.0 + p / c["cell_pstar"]) / LN10,
                         c["cell_a"] + c["cell_b"] * lp)
        edge_ok = c["edge_ok"]
        s = tf.where(edge_ok, c["edge_A"] * p ** c["edge_alpha"], 0.5 * c["z_max"])
        tf.debugging.assert_less(tf.boolean_mask(s, edge_ok), c["z_max"], message="electron edge not below d_m")
        tf.debugging.assert_positive(tf.boolean_mask(s, edge_ok), message="electron edge not positive")
        d, s3 = c["zeff"][None], s[..., None]
        zeta = tf.where(d < s3, d / s3, tf.where(d < c["z_max"], 1.0 + (d - s3) / (c["z_max"] - s3),
                                                 2.0 + (d - c["z_max"])))
        xp, fp = c["table_zeta"], c["table_profile"]
        xq = tf.broadcast_to(zeta[None], tf.concat([tf.shape(xp)[:3], tf.shape(zeta)[-1:]], 0))
        k = tf.clip_by_value(tf.searchsorted(xp, xq, side="right") - 1, 0, xp.shape[-1] - 2)
        x0, x1 = tf.gather(xp, k, batch_dims=3), tf.gather(xp, k + 1, batch_dims=3)
        f0, f1 = tf.gather(fp, k, batch_dims=3), tf.gather(fp, k + 1, batch_dims=3)
        v = f0 + (xq - x0) * (f1 - f0) / (x1 - x0)
        v = tf.where(xq <= xp[..., :1], fp[..., :1], v)
        v = tf.where(xq >= xp[..., -1:], fp[..., -1:], v)
        one = tf.constant(1.0, tf.float64)
        line = tf.einsum("n,ntjr->tjr", tf.linalg.matvec(c["line_weights"], tf.stack([one, lp]), transpose_a=True), v)
        regions = [(d > c["z_tol"]) & (d <= s3), (d > s3) & c["low"][None]]
        aligned = tf.zeros_like(line)
        for i, region in enumerate(regions):
            coef = tf.einsum("kn,ntjr->ktjr", c["region_pinv"][i], v)
            h = tf.stack([one, lp, -tf.math.log(1.0 + p / c["region_pstar"][i]) / LN10])
            aligned = tf.where(region, tf.where(coef[2] < 0, line, tf.einsum("k,ktjr->tjr", h, coef)), aligned)
        gap = tf.gather(tf.reshape(aligned, [self.n_frames, -1]), self.gap_cells, axis=1)
        gap = tf.where(tf.gather(edge_ok, self.gap_column, axis=1), gap, tf.gather(field, self.gap_valid, axis=1))
        return tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(field), self.gap_valid[:, None],
                                                        tf.transpose(gap)))

    def call(self, probes):
        c = self.c
        mu, sigma = tf.cast(c["target_mu"], tf.float32), tf.cast(c["target_sigma"], tf.float32)
        n_cells = self.grid_shape[0] * self.grid_shape[1]
        shape = (self.n_frames,) + self.grid_shape + (1,)

        def one_case(p):
            z = (tf.cast(self.physics_guided_field(p), tf.float32) - mu) / sigma
            grid = tf.scatter_nd(c["valid_index"][:, None], tf.transpose(z), [n_cells, self.n_frames])
            return tf.reshape(tf.transpose(grid), shape)

        return tf.map_fn(one_case, self.operating_condition(probes), fn_output_signature=tf.TensorSpec(shape, tf.float32))


class UnitGain(keras.layers.Layer):

    def build(self, input_shape):
        self.gain = self.add_weight(name="gain", shape=[], initializer="zeros", trainable=True)

    def call(self, x):
        return tf.cast(self.gain, x.dtype) * x


def conv3d(x, channels, kernel, strides, name, causal, groups=1, use_bias=False, activation=None):
    past = kernel[0] - 1 if causal else 0
    if past:
        x = keras.layers.ZeroPadding3D(((past, 0), (0, 0), (0, 0)), name=f"{name}_past_frames")(x)
    x = keras.layers.Conv3D(channels, kernel, strides=strides, padding="same", groups=groups, use_bias=use_bias,
                            kernel_initializer="he_normal", activation=activation, name=name)(x)
    if past:
        x = keras.layers.Cropping3D(((past // 2, past - past // 2), (0, 0), (0, 0)), name=f"{name}_current_frames")(x)
    return x


def batch_norm(x, name, synchronized=True):
    return keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3, synchronized=synchronized, name=name)(x)


def squeeze_channels(expand):
    value = 0.25 * expand
    channels = max(8, int(value + 4) // 8 * 8)
    return channels + 8 if channels < 0.9 * value else channels


def frame_average(causal):
    if causal:
        return lambda t: (tf.cumsum(tf.reduce_mean(t, axis=[2, 3], keepdims=True), axis=1)
                          / tf.reshape(tf.cast(tf.range(1, tf.shape(t)[1] + 1), t.dtype), [1, -1, 1, 1, 1]))
    return lambda t: (tf.reduce_sum(t, axis=(1, 2, 3), keepdims=True) / tf.cast(tf.shape(t)[2] * tf.shape(t)[3], t.dtype)
                      / tf.cast(tf.shape(t)[1], t.dtype))


def unit(x, out_channels, expand, kernel, strides, causal, name):
    shortcut = x
    x = conv3d(x, expand, (1, 1, 1), (1, 1, 1), f"{name}_expansion_conv", causal)
    x = keras.layers.Activation("swish", name=f"{name}_expansion_swish")(batch_norm(x, f"{name}_expansion_bn"))
    x = conv3d(x, expand, kernel, strides, f"{name}_depthwise_conv", causal, groups=expand)
    x = keras.layers.Activation("swish", name=f"{name}_depthwise_swish")(batch_norm(x, f"{name}_depthwise_bn"))
    s = keras.layers.Lambda(frame_average(causal), name=f"{name}_squeeze_average")(x)
    s = conv3d(s, squeeze_channels(expand), (1, 1, 1), (1, 1, 1), f"{name}_squeeze_reduce", causal, use_bias=True,
               activation="swish")
    s = conv3d(s, expand, (1, 1, 1), (1, 1, 1), f"{name}_squeeze_expand", causal, use_bias=True, activation="sigmoid")
    x = keras.layers.Multiply(name=f"{name}_squeeze_excite")([s, x])
    x = batch_norm(conv3d(x, out_channels, (1, 1, 1), (1, 1, 1), f"{name}_projection_conv", causal),
                   f"{name}_projection_bn")
    x = UnitGain(name=f"{name}_gain")(x)
    strided = any(stride > 1 for stride in strides)
    if strided or shortcut.shape[-1] != out_channels:
        if strided:
            shortcut = keras.layers.AveragePooling3D((1, 3, 3), strides=(1, 2, 2), padding="same",
                                                     name=f"{name}_shortcut_pool")(shortcut)
        shortcut = batch_norm(conv3d(shortcut, out_channels, (1, 1, 1), (1, 1, 1), f"{name}_shortcut_conv", False),
                              f"{name}_shortcut_bn", synchronized=False)
    return keras.layers.Add(name=f"{name}_residual")([x, shortcut])


def stem(x, causal):
    x = batch_norm(conv3d(x, 8, (1, 3, 3), (1, 2, 2), "stem_conv", causal), "stem_bn")
    return keras.layers.Activation("swish", name="stem_swish")(x)


def block(x, index, out_channels, units, causal):
    for u, (expand, kernel, strides) in enumerate(units):
        x = unit(x, out_channels, expand, kernel, strides, causal, f"block{index}_unit{u}")
    return x


def encoder(probes, coefficient_file, causal):
    physics_guided_field = PhysicsGuidedUpsampling(coefficient_file, name="physics_guided_upsampling")(probes)
    x = stem(probes, causal)
    x = block(x, 0, 8, [(24, (1, 5, 5), (1, 2, 2))], causal)
    x = block(x, 1, 32, [(80, (3, 3, 3), (1, 2, 2)), (80, (3, 3, 3), (1, 1, 1)), (80, (3, 3, 3), (1, 1, 1))], causal)
    x = block(x, 2, 56, [(184, (5, 3, 3), (1, 2, 2)), (112, (3, 3, 3), (1, 1, 1)), (184, (3, 3, 3), (1, 1, 1))], causal)
    x = block(x, 3, 56, [(184, (5, 3, 3), (1, 1, 1)), (184, (3, 3, 3), (1, 1, 1)), (184, (3, 3, 3), (1, 1, 1)),
                         (184, (3, 3, 3), (1, 1, 1))], causal)
    return x, physics_guided_field


def decoder(features, physics_guided_field, chamber, grid_height, grid_width):
    x = keras.layers.ConvLSTM2D(64, (3, 3), padding="same", activation="relu", return_sequences=True,
                                name="convlstm2d_1")(features)
    for i, filters in enumerate((32, 16, 16, 8)):
        x = keras.layers.Conv3DTranspose(filters, (3, 3, 3), strides=(1, 2, 2), padding="same",
                                         name=f"conv3d_transpose_{i + 1}")(x)
    x = keras.layers.Cropping3D(((0, 0), (0, x.shape[2] - grid_height), (0, x.shape[3] - grid_width)),
                                name="crop_to_grid")(x)
    x = keras.layers.Concatenate(axis=-1, name="physics_guided_concat")([x, physics_guided_field])
    for i, filters in enumerate((64, 32, 16)):
        x = keras.layers.ConvLSTM2D(filters, (3, 3), padding="same", activation="relu", return_sequences=True,
                                    name=f"convlstm2d_{i + 2}")(x)
    x = keras.layers.ConvLSTM2D(1, (3, 3), padding="same", activation="linear", return_sequences=True,
                                name="convlstm2d_5")(x)
    field = keras.layers.Add(name="output_field")([physics_guided_field, x])
    return keras.layers.Lambda(lambda t: t * chamber, name="chamber_mask")(field)


def weight_tensor(loss_weight):
    w = tf.constant(loss_weight, dtype=tf.float32)
    return tf.reshape(w, [1, 1, int(w.shape[0]), int(w.shape[1]), 1])


def masked_mse(loss_weight):
    w = weight_tensor(loss_weight)

    def loss(y_true, y_pred):
        return tf.reduce_sum(w * tf.square(y_pred - y_true)) / (tf.reduce_sum(w * tf.ones_like(y_true)) + 1e-8)

    return loss


class MaskedRMSE(keras.metrics.Metric):

    def __init__(self, loss_weight, name="root_mean_squared_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.w = weight_tensor(loss_weight)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.sse.assign_add(tf.reduce_sum(self.w * tf.square(y_pred - y_true)))
        self.count.assign_add(tf.reduce_sum(self.w * tf.ones_like(y_true)))

    def result(self):
        return tf.sqrt(self.sse / (self.count + 1e-8))

    def reset_state(self):
        self.sse.assign(0.0)
        self.count.assign(0.0)


def build_spire_reconstruction_model(coefficient_file, loss_weight, causal_encoder=False, n_frames=20,
                                     grid_height=120, grid_width=160):
    probes = keras.Input((n_frames, grid_height, grid_width, 3), name="probes")
    features, physics_guided_field = encoder(probes, coefficient_file, causal_encoder)
    chamber = weight_tensor((np.asarray(loss_weight) > 0).astype(np.float32))
    field = decoder(features, physics_guided_field, chamber, grid_height, grid_width)
    model = keras.Model(inputs=probes, outputs=field, name="spire_reconstruction")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
                  loss=masked_mse(loss_weight), metrics=[MaskedRMSE(loss_weight)])
    return model
