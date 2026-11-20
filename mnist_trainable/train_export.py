import numpy as np
import tensorflow as tf

IMG = 28
CLASSES = 10

# ---------- 1. Предобучение CNN на MNIST с лёгкой аугментацией ----------
aug_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG, IMG, 1)),
    tf.keras.layers.RandomRotation(0.06),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.Conv2D(16, 3, activation='relu', name='conv1'),   # 26x26x16
    tf.keras.layers.MaxPooling2D(2),                                  # 13x13x16
    tf.keras.layers.Conv2D(32, 3, activation='relu', name='conv2'),   # 11x11x32
    tf.keras.layers.MaxPooling2D(2),                                  # 5x5x32
    tf.keras.layers.Flatten(),                                        # 800
    tf.keras.layers.Dense(64, activation='relu', name='dense1'),      # 64
    tf.keras.layers.Dense(CLASSES, name='logits'),                    # 10
])

aug_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
x_tr = (x_tr[..., None] / 255.0).astype("float32")
x_te = (x_te[..., None] / 255.0).astype("float32")
y_tr = tf.keras.utils.to_categorical(y_tr, CLASSES).astype("float32")
y_te = tf.keras.utils.to_categorical(y_te, CLASSES).astype("float32")

aug_model.fit(x_tr, y_tr, batch_size=128, epochs=6, verbose=1)
_, acc = aug_model.evaluate(x_te, y_te, verbose=0)
print(f"Pretrained CNN test accuracy: {acc:.4f}")

# Извлекаем обученные веса слоёв
c1_k, c1_b = [w.numpy().astype("float32") for w in aug_model.get_layer('conv1').weights]
c2_k, c2_b = [w.numpy().astype("float32") for w in aug_model.get_layer('conv2').weights]
w1_init, b1_init = [w.numpy().astype("float32") for w in aug_model.get_layer('dense1').weights]
w2_init, b2_init = [w.numpy().astype("float32") for w in aug_model.get_layer('logits').weights]

# ---------- 2. Экспорт CNN в tf.Module (замороженные свёртки + обучаемая голова) ----------
class TrainableCNN(tf.Module):
    def __init__(self, c1_k, c1_b, c2_k, c2_b, w1, b1, w2, b2):
        super().__init__()
        # Замороженный экстрактор признаков
        self.c1_k = tf.constant(c1_k, dtype=tf.float32)
        self.c1_b = tf.constant(c1_b, dtype=tf.float32)
        self.c2_k = tf.constant(c2_k, dtype=tf.float32)
        self.c2_b = tf.constant(c2_b, dtype=tf.float32)

        # Начальные константы головы (для init_weights и сброса)
        self._w1_const = tf.constant(w1, dtype=tf.float32)
        self._b1_const = tf.constant(b1, dtype=tf.float32)
        self._w2_const = tf.constant(w2, dtype=tf.float32)
        self._b2_const = tf.constant(b2, dtype=tf.float32)

        # Обучаемые переменные классификатора (800 -> 64 -> 10)
        self.w1 = tf.Variable(w1, name="w1")
        self.b1 = tf.Variable(b1, name="b1")
        self.w2 = tf.Variable(w2, name="w2")
        self.b2 = tf.Variable(b2, name="b2")

    def _extract_features(self, x):
        x4 = tf.reshape(x, [-1, IMG, IMG, 1])
        h1 = tf.nn.relu(tf.nn.conv2d(x4, self.c1_k, strides=1, padding="VALID") + self.c1_b)
        p1 = tf.nn.max_pool2d(h1, ksize=2, strides=2, padding="VALID")
        h2 = tf.nn.relu(tf.nn.conv2d(p1, self.c2_k, strides=1, padding="VALID") + self.c2_b)
        p2 = tf.nn.max_pool2d(h2, ksize=2, strides=2, padding="VALID")
        return tf.reshape(p2, [-1, 5 * 5 * 32])

    def _forward(self, x):
        feats = self._extract_features(x)
        h = tf.nn.relu(tf.matmul(feats, self.w1) + self.b1)
        return tf.matmul(h, self.w2) + self.b2

    @tf.function(input_signature=[tf.TensorSpec([1], tf.float32, name='trigger')])
    def init_weights(self, trigger):
        self.w1.assign(self._w1_const)
        self.b1.assign(self._b1_const)
        self.w2.assign(self._w2_const)
        self.b2.assign(self._b2_const)
        return {"ok": trigger + 1.0}

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG, IMG], tf.float32, name='x'),
        tf.TensorSpec([None, CLASSES], tf.float32, name='y'),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            logits = self._forward(x)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            )
        vars_list = [self.w1, self.b1, self.w2, self.b2]
        grads = tape.gradient(loss, vars_list)
        for v, g in zip(vars_list, grads):
            v.assign_sub(0.05 * g)
        return {"loss": tf.reshape(loss, [1])}

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG, IMG], tf.float32, name='x'),
    ])
    def infer(self, x):
        logits = self._forward(x)
        return {"output": tf.nn.softmax(logits, axis=-1)}

    @tf.function(input_signature=[tf.TensorSpec([1], tf.string, name='checkpoint_path')])
    def save(self, checkpoint_path):
        path = checkpoint_path[0]
        names = ["w1", "b1", "w2", "b2"]
        values = [self.w1.read_value(), self.b1.read_value(),
                  self.w2.read_value(), self.b2.read_value()]
        tf.raw_ops.Save(filename=path, tensor_names=names, data=values, name='save')
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec([1], tf.string, name='checkpoint_path')])
    def restore(self, checkpoint_path):
        path = checkpoint_path[0]
        restored = {}
        for name, var in [("w1", self.w1), ("b1", self.b1), ("w2", self.w2), ("b2", self.b2)]:
            val = tf.raw_ops.Restore(file_pattern=path, tensor_name=name, dt=tf.float32, name='restore')
            val = tf.reshape(val, var.shape)
            var.assign(val)
            restored[name] = val
        return restored

m = TrainableCNN(c1_k, c1_b, c2_k, c2_b, w1_init, b1_init, w2_init, b2_init)

tf.saved_model.save(m, "saved_model", signatures={
    'init_weights': m.init_weights.get_concrete_function(),
    'train':        m.train.get_concrete_function(),
    'infer':        m.infer.get_concrete_function(),
    'save':         m.save.get_concrete_function(),
    'restore':      m.restore.get_concrete_function(),
})

conv = tf.lite.TFLiteConverter.from_saved_model("saved_model")
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
conv.experimental_enable_resource_variables = True
tflite = conv.convert()
open("mnist_trainable.tflite", "wb").write(tflite)
print(f"Saved CNN mnist_trainable.tflite ({len(tflite)/1024:.1f} KB)")

# ---------- 3. Автопроверка всех 5 сигнатур ----------
interp = tf.lite.Interpreter(model_content=tflite)
interp.get_signature_runner("init_weights")(trigger=np.array([1.0], dtype=np.float32))
test_x = x_te[:1, :, :, 0]
test_y = y_te[:1]
res_infer = interp.get_signature_runner("infer")(x=test_x)
print("Self-test infer OK, pred:", int(np.argmax(res_infer["output"])))
res_train = interp.get_signature_runner("train")(x=test_x, y=test_y)
print("Self-test train OK, loss:", float(res_train["loss"][0]))
interp.get_signature_runner("save")(checkpoint_path=np.array(["/tmp/test_cnn.ckpt"], dtype=object))
interp.get_signature_runner("restore")(checkpoint_path=np.array(["/tmp/test_cnn.ckpt"], dtype=object))
print("Self-test CNN all 5 signatures OK!")
