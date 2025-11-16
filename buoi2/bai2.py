import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# ==========================
# 1. LOAD + CHỌN 5 LỚP × 1000 ẢNH
# ==========================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

selected_classes = [0, 1, 2, 3, 4]   # chọn 5 lớp bất kỳ
y_train = y_train.flatten()

x_filtered, y_filtered = [], []
for cls in selected_classes:
    idx = np.where(y_train == cls)[0][:1000]
    x_filtered.append(x_train[idx])
    y_filtered.append(y_train[idx])

x_train = np.concatenate(x_filtered)
y_train = np.concatenate(y_filtered)

x_train = x_train.astype("float32") / 255.0
y_train = to_categorical(y_train, num_classes=10)

x_test = x_test.astype("float32") / 255.0
y_test = to_categorical(y_test, num_classes=10)

# ==========================
# 2. HIỂN THỊ ẢNH GỐC
# ==========================
plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i])
    plt.axis("off")
plt.suptitle("Một số ảnh gốc (Original Images)")
plt.savefig("original_images_examples.png")
plt.show()

# ==========================
# 3. HÀM XÂY DỰNG MODEL CNN
# ==========================
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================
# 4. TRAIN 3 LẦN LẤY TRUNG BÌNH
# ==========================
results = []

for run in range(3):
    print(f"\n===== RUN {run+1}/3 (Original Data) =====")
    model = build_model()
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=10,
        validation_split=0.2,
        verbose=1
    )
    results.append(history.history)

# ==========================
# 5. TÍNH TRUNG BÌNH
# ==========================
mean_val_acc = np.mean([r['val_accuracy'][-1] for r in results])
mean_val_loss = np.mean([r['val_loss'][-1] for r in results])

print("\n===== KẾT QUẢ TRUNG BÌNH — ORIGINAL DATA =====")
print("Validation Accuracy:", round(mean_val_acc, 4))
print("Validation Loss:", round(mean_val_loss, 4))

# ==========================
# 6. LƯU FILE CSV + EXCEL
# ==========================
df = pd.DataFrame({
    "Run": [1, 2, 3],
    "Val_Accuracy": [r['val_accuracy'][-1] for r in results],
    "Val_Loss": [r['val_loss'][-1] for r in results]
})
df.loc["Mean"] = ["-", mean_val_acc, mean_val_loss]

df.to_csv("result_original.csv", index=False)
df.to_excel("result_original.xlsx", index=False)

print("\n✔ Đã lưu file: result_original.csv / result_original.xlsx")

# ==========================
# 7. VẼ BIỂU ĐỒ ACC / LOSS
# ==========================
plt.figure(figsize=(8,5))
for r in results:
    plt.plot(r["accuracy"])
plt.title("Training Accuracy (Original Data)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("original_accuracy_curve.png")
plt.show()

plt.figure(figsize=(8,5))
for r in results:
    plt.plot(r["loss"])
plt.title("Training Loss (Original Data)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("original_loss_curve.png")
plt.show()
