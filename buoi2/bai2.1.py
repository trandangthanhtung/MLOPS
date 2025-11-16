# train_augmented_fixed.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------
# 1. Load + chọn 5 lớp x1000
# -------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()

selected_classes = [0,1,2,3,4]   # bạn có thể đổi
x_filtered = []
y_filtered = []
for cls in selected_classes:
    idx = np.where(y_train == cls)[0][:1000]
    x_filtered.append(x_train[idx])
    y_filtered.append(y_train[idx])

x_train = np.concatenate(x_filtered)   # shape (5000, 32, 32, 3), dtype=uint8
y_train = np.concatenate(y_filtered)

# chuẩn hóa labels cho one-hot nếu cần (ở đây dùng categorical)
y_train_cat = to_categorical(y_train, num_classes=10)

# chuẩn hóa test (để validation)
x_test = x_test.astype("float32") / 255.0
y_test_cat = to_categorical(y_test.flatten(), num_classes=10)

print("Train shape:", x_train.shape, "Train labels:", y_train.shape)

# -------------------------
# 2. Datagen: geometric transforms (safe)
# -------------------------
# Không đặt brightness_range ở đây để tránh các vấn đề với 0-1 scaling
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.18,
    horizontal_flip=True,
    shear_range=0.12,
    fill_mode="nearest"
)
# NOTE: datagen expects floats; we'll feed normalized images when using .flow

# -------------------------
# 3. Hàm điều chỉnh brightness/contrast AN TOÀN (áp sau generator)
# -------------------------
def adjust_brightness_contrast_batch(img_batch, brightness_delta=0.15, contrast_delta=0.20):
    """
    img_batch: numpy array float32 in [0,1], shape (N, H, W, C)
    returns clipped array in [0,1]
    """
    out = np.empty_like(img_batch)
    for i, img in enumerate(img_batch):
        # brightness: multiply by factor near 1
        b_factor = 1.0 + np.random.uniform(-brightness_delta, brightness_delta)
        img_b = img * b_factor

        # contrast: scale around mean
        c_factor = 1.0 + np.random.uniform(-contrast_delta, contrast_delta)
        mean = img_b.mean(axis=(0,1), keepdims=True)
        img_c = (img_b - mean) * c_factor + mean

        img_c = np.clip(img_c, 0.0, 1.0)
        out[i] = img_c
    return out

# -------------------------
# 4. Hiển thị ví dụ BEFORE / AFTER (an toàn)
# -------------------------
def show_examples(n=5):
    # lấy n mẫu
    samples = x_train[:n].astype("float32") / 255.0  # normalize for datagen
    aug_iter = datagen.flow(samples, batch_size=n, shuffle=False)
    aug_batch = next(aug_iter)                      # float32 in [0,1] (approx)
    # áp dụng brightness/contrast an toàn
    aug_batch = adjust_brightness_contrast_batch(aug_batch, brightness_delta=0.12, contrast_delta=0.18)

    plt.figure(figsize=(12,5))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(samples[i])
        plt.axis("off")
        if i==0:
            plt.title("Original (normalized)")

        plt.subplot(2, n, n + i + 1)
        plt.imshow(aug_batch[i])
        plt.axis("off")
        if i==0:
            plt.title("Augmented (safe)")
    plt.suptitle("Before (top) / After (bottom) augmentation")
    plt.show()

show_examples(5)

# -------------------------
# 5. Tạo full augmented dataset (x_train_aug)
#    - Ở đây mình tạo bằng 1 epoch kích thước=original (1-to-1)
#    - Nếu muốn mở rộng dữ liệu nhiều hơn, lặp thêm lần
# -------------------------
def create_augmented_dataset(x_raw):
    # x_raw: uint8 or float in [0,1]? We'll convert to float in [0,1]
    x_norm = x_raw.astype("float32") / 255.0
    batch_size = 128
    n = x_norm.shape[0]
    aug_out = []
    for i in range(0, n, batch_size):
        batch = x_norm[i:i+batch_size]
        aug_iter = datagen.flow(batch, batch_size=batch.shape[0], shuffle=False)
        aug_batch = next(aug_iter)                       # in [0,1]
        aug_batch = adjust_brightness_contrast_batch(aug_batch,
                                                     brightness_delta=0.12,
                                                     contrast_delta=0.18)
        aug_out.append(aug_batch)
    aug_out = np.vstack(aug_out)
    return aug_out

print("Creating augmented dataset (this may take a minute)...")
x_train_aug = create_augmented_dataset(x_train)   # float32 in [0,1]
print("Augmented dataset shape:", x_train_aug.shape)

# You can combine original+augmented if you want a larger dataset:
x_train_combined = np.concatenate([x_train.astype("float32")/255.0, x_train_aug], axis=0)
y_train_combined = np.concatenate([y_train_cat, y_train_cat], axis=0)   # doubled labels

print("Combined train shape:", x_train_combined.shape, y_train_combined.shape)

# -------------------------
# 6. Model (simple CNN)
# -------------------------
def build_model(num_classes=10):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# 7. Train 3 runs (using combined dataset) and collect histories
# -------------------------
runs = 3
histories = []
for run in range(runs):
    print(f"\n=== TRAIN RUN {run+1}/{runs} ===")
    model = build_model(num_classes=10)
    history = model.fit(
        x_train_combined, y_train_combined,
        validation_data=(x_test, y_test_cat),
        epochs=12,
        batch_size=64,
        verbose=1
    )
    histories.append(history.history)

# -------------------------
# 8. Save results CSV/Excel + plot acc/loss
# -------------------------
val_accs = [h['val_accuracy'][-1] for h in histories]
val_losses = [h['val_loss'][-1] for h in histories]
train_accs = [h['accuracy'][-1] for h in histories]
train_losses = [h['loss'][-1] for h in histories]

df = pd.DataFrame({
    'run': list(range(1, runs+1)),
    'train_acc': train_accs,
    'train_loss': train_losses,
    'val_acc': val_accs,
    'val_loss': val_losses
})
df.loc['mean'] = df.mean(numeric_only=True)
df.to_csv('results_augmented_fixed.csv', index=False)
df.to_excel('results_augmented_fixed.xlsx', index=False)
print("Saved results_augmented_fixed.csv / xlsx")

# plot accuracy curves (all runs)
plt.figure(figsize=(8,5))
for h in histories:
    plt.plot(h['accuracy'], alpha=0.6)
plt.title("Train accuracy (all runs)")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("aug_train_accuracy_allruns.png")
plt.show()

plt.figure(figsize=(8,5))
for h in histories:
    plt.plot(h['val_accuracy'], alpha=0.6)
plt.title("Val accuracy (all runs)")
plt.xlabel("epoch")
plt.ylabel("val_accuracy")
plt.savefig("aug_val_accuracy_allruns.png")
plt.show()

# -------------------------
# DONE
# -------------------------
print("Finished.")
