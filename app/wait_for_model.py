import os
import time

flag = "models/MODEL_READY"

print("⏳ Waiting for trained model...")
while not os.path.exists(flag):
    time.sleep(5)

print("✅ Model ready")