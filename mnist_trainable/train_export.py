import numpy as np
import tensorflow as tf

IMG = 28
CLASSES = 10

# ---------- 1. Предобучение на MNIST ----------
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG, IMG)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(CLASSES),
])
keras_model.compile(
    optimizer=tf.keras.optimizers.SGD(0.05),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

(x_tr, y_tr), _ = tf.keras.datasets.mnist.load_data()
x_tr = (x_tr / 255.0).astype("float32")
y_tr = tf.keras.utils.to_categorical(y_tr, CLASSES).astype("float32")

keras_model.fit(x_tr, y_tr, batch_size=128, epochs=5, verbose=1)
_, acc = keras_model.evaluate(x_tr[:5000], y_tr[:5000], verbose=0)
print(f"Pretrained accuracy: {acc:.4f}")

w1_init, b1_init, w2_init, b2_init = [w.numpy().astype("float32") for w in keras_model.weights]

# ---------- 2. Чистый tf.Module с tf.Variable для TFLite On-Device Training ----------
class TrainableModel(tf.Module):
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.w1 = tf.Variable(w1, name="w1")
        self.b1 = tf.Variable(b1, name="b1")
        self.w2 = tf.Variable(w2, name="w2")
        self.b2 = tf.Variable(b2, name="b2")

    def _forward(self, x):
        flat = tf.reshape(x, [-1, IMG * IMG])
        h = tf.nn.relu(tf.matmul(flat, self.w1) + self.b1)
        return tf.matmul(h, self.w2) + self.b2

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

m = TrainableModel(w1_init, b1_init, w2_init, b2_init)

tf.saved_model.save(m, "saved_model", signatures={
    'train':   m.train.get_concrete_function(),
    'infer':   m.infer.get_concrete_function(),
    'save':    m.save.get_concrete_function(),
    'restore': m.restore.get_concrete_function(),
})

conv = tf.lite.TFLiteConverter.from_saved_model("saved_model")
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
conv.experimental_enable_resource_variables = True
tflite = conv.convert()
open("mnist_trainable.tflite", "wb").write(tflite)
print(f"Saved mnist_trainable.tflite ({len(tflite)/1024:.1f} KB)")

# ---------- 3. Автопроверка TFLite Interpreter при сборке ----------
interp = tf.lite.Interpreter(model_content=tflite)
interp.allocate_tensors()
infer_fn = interp.get_signature_runner("infer")
train_fn = interp.get_signature_runner("train")
save_fn = interp.get_signature_runner("save")
restore_fn = interp.get_signature_runner("restore")

test_x = x_tr[:1]
test_y = y_tr[:1]
res_infer = infer_fn(x=test_x)
print("Self-test infer OK, shape:", res_infer["output"].shape, "pred:", int(np.argmax(res_infer["output"])))
res_train = train_fn(x=test_x, y=test_y)
print("Self-test train OK, loss:", float(res_train["loss"][0]))
save_fn(checkpoint_path=np.array(["/tmp/test_digit.ckpt"], dtype=object))
restore_fn(checkpoint_path=np.array(["/tmp/test_digit.ckpt"], dtype=object))
print("Self-test save & restore OK!")
