import numpy as np
import tensorflow as tf

IMG = 28
CLASSES = 10

# ---------- Обучаемая модель ----------
class TrainableModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG, IMG), name='flatten'),
            tf.keras.layers.Dense(128, activation='relu', name='dense1'),
            tf.keras.layers.Dense(CLASSES, name='logits'),
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG, IMG], tf.float32, name='x'),
        tf.TensorSpec([None, CLASSES], tf.float32, name='y'),
    ])
    def train(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.model.loss(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        for var, grad in zip(self.model.trainable_variables, grads):
            var.assign_sub(0.01 * grad)
        return {"loss": tf.reshape(loss, [1])}

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMG, IMG], tf.float32, name='x'),
    ])
    def infer(self, x):
        logits = self.model(x, training=False)
        return {"output": tf.nn.softmax(logits, axis=-1)}

    @tf.function(input_signature=[tf.TensorSpec([1], tf.string, name='checkpoint_path')])
    def save(self, checkpoint_path):
        path = checkpoint_path[0]
        names = [f"w_{i}" for i, _ in enumerate(self.model.weights)]
        values = [tf.convert_to_tensor(w) for w in self.model.weights]
        tf.raw_ops.Save(filename=path, tensor_names=names,
                        data=values, name='save')
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec([1], tf.string, name='checkpoint_path')])
    def restore(self, checkpoint_path):
        path = checkpoint_path[0]
        out = {}
        for i, var in enumerate(self.model.weights):
            t_name = f"w_{i}"
            val = tf.raw_ops.Restore(file_pattern=path,
                                     tensor_name=t_name,
                                     dt=tf.as_dtype(var.dtype), name='restore')
            val = tf.reshape(val, var.shape)
            var.assign(val)
            out[t_name] = val
        return out


if __name__ == "__main__":
    # ---------- Предобучение на MNIST ----------
    m = TrainableModel()
    (x_tr, y_tr), _ = tf.keras.datasets.mnist.load_data()
    x_tr = (x_tr / 255.0).astype("float32")
    y_tr = tf.keras.utils.to_categorical(y_tr, CLASSES).astype("float32")

    m.model.fit(x_tr, y_tr, batch_size=128, epochs=3, verbose=1)
    eval_res = m.model.evaluate(x_tr[:5000], y_tr[:5000], verbose=0)
    acc = eval_res[1] if isinstance(eval_res, (list, tuple)) and len(eval_res) > 1 else eval_res
    print(f"Pretrained accuracy: {acc:.4f}")

    # ---------- Экспорт SavedModel ----------
    tf.saved_model.save(m, "saved_model", signatures={
        'train':   m.train.get_concrete_function(),
        'infer':   m.infer.get_concrete_function(),
        'save':    m.save.get_concrete_function(),
        'restore': m.restore.get_concrete_function(),
    })

    # ---------- Конвертация в TFLite ----------
    conv = tf.lite.TFLiteConverter.from_saved_model("saved_model")
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    conv.experimental_enable_resource_variables = True
    tflite = conv.convert()
    open("mnist_trainable.tflite", "wb").write(tflite)
    print(f"Saved mnist_trainable.tflite ({len(tflite)/1024:.1f} KB)")
