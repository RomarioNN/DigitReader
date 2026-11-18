# DigitReader — Android-приложение «Распознавание цифры» с дообучением на устройстве

Готовый Android-проект по вашей инструкции с дообучаемой моделью TensorFlow Lite (`train`, `infer`, `save`, `restore`).

## Важные улучшения в коде для стабильной работы без вылетов:
1. **Зависимость `tensorflow-lite-select-tf-ops:2.16.1`** добавлена в `app/build.gradle.kts`, так как модель экспортируется с `OpsSet.SELECT_TF_OPS` и `experimental_enable_resource_variables = True` (для поддержки операций `Save`, `Restore` и обучаемых переменных).
2. **Приведение размерности входа `[1, 28, 28]`** в `DigitClassifier.kt` (`toBatch28x28`), чтобы `FloatArray(784)` точно соответствовал сигнатуре `tf.TensorSpec([None, 28, 28], tf.float32)` из `train_export.py` и не вызывал ошибку `Input tensor shape mismatch`.
3. **Метрика `metrics=['accuracy']`** добавлена в `model.compile(...)` в `train_export.py`, чтобы `m.model.evaluate(...)[1]` корректно возвращал точность.

## Как получить `.apk` за 2 минуты:

### Вариант А: Облачная автосборка через GitHub Actions (без установки Android Studio)
1. Создайте репозиторий на GitHub и загрузите туда содержимое этого архива.
2. Откройте вкладку **Actions** -> выберите **Build DigitReader APK** -> нажмите **Run workflow**.
3. GitHub автоматически обучит `mnist_trainable.tflite` на MNIST, положит его в `app/src/main/assets/`, соберёт APK и прикрепит готовый **`DigitReader-debug-apk.zip` (`app-debug.apk`)** для скачивания на телефон.

### Вариант Б: Локальная сборка в Android Studio
1. Запустите генерацию модели:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install tensorflow==2.16.1 numpy
   python mnist_trainable/train_export.py
   mkdir -p app/src/main/assets
   cp mnist_trainable.tflite app/src/main/assets/
   ```
2. Откройте папку проекта в **Android Studio Hedgehog (2023.1+)**.
3. Нажмите **Build -> Build Bundle(s) / APK(s) -> Build APK(s)** или запустите на телефоне кнопкой **Run ▶️**.
