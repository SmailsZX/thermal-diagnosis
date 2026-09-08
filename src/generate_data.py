import os
import json
import random
from collections import Counter

import cv2
import numpy as np


def create_realistic_training_dataset(num_images=500, save_dir="training_thermal_images"):
    os.makedirs(save_dir, exist_ok=True)
    labels = {}
    print(f"🖼️ Создаю {num_images} РЕАЛИСТИЧНЫХ термограмм...")
    for i in range(num_images):
        img = np.random.normal(80, 20, (224, 224))
        cv2.rectangle(img, (50, 50), (174, 174), 100, -1)
        texture = np.random.normal(0, 10, (124, 124))
        img[50:174, 50:174] += texture
        state = random.choice(['normal', 'overheat', 'fault'])
        if state == 'overheat':
            cv2.circle(img, (112, 112), 40, 180, -1)
            for y in range(50, 174):
                for x in range(50, 174):
                    dist = np.sqrt((x-112)**2 + (y-112)**2)
                    if dist < 40:
                        heat = 180 * (1 - dist/40)
                        img[y, x] = min(255, img[y, x] + heat)
            label = 1
        elif state == 'fault':
            hot_x, hot_y = random.randint(60, 80), random.randint(60, 80)
            cv2.circle(img, (hot_x, hot_y), 15, 220, -1)
            label = 2
        else:
            label = 0
        img = np.clip(img + np.random.normal(0, 5, (224, 224)), 0, 255)
        img = img.astype(np.uint8)
        filename = f"thermal_{i:04d}.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, img)
        labels[filepath] = label
        if (i + 1) % 100 == 0:
            print(f"   Создано {i+1}/{num_images}...")
    with open("training_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    stats = Counter(labels.values())
    print(f"\n✅ Создан РЕАЛИСТИЧНЫЙ датасет:")
    print(f"   Норма (0): {stats[0]} шт.")
    print(f"   Перегрев (1): {stats[1]} шт.")
    print(f"   Неисправность (2): {stats[2]} шт.")
    return "training_labels.json", save_dir


def generate_realistic_thermal_images(num_images=200, images_dir="real_thermal_dataset"):
    os.makedirs(images_dir, exist_ok=True)
    labels = {}
    print(f"🖼️ Генерирую {num_images} физически корректных термограмм...")
    for i in range(num_images):
        img_size = 224
        background_temp = np.random.normal(25, 5)
        img = np.ones((img_size, img_size)) * background_temp
        equipment_zone = np.zeros((img_size, img_size))
        cv2.rectangle(equipment_zone, (60, 60), (164, 164), 1, -1)
        state = np.random.choice(['normal', 'overheat', 'fault'], p=[0.5, 0.25, 0.25])
        if state == 'normal':
            equipment_temp = background_temp + np.random.uniform(5, 10)
            img[equipment_zone == 1] = equipment_temp
            label = 0
        elif state == 'overheat':
            equipment_temp = background_temp + np.random.uniform(15, 25)
            img[equipment_zone == 1] = equipment_temp
            center_hot = np.random.uniform(30, 40)
            cv2.circle(img, (112, 112), 30, center_hot, -1)
            label = 1
        else:
            equipment_temp = background_temp + np.random.uniform(5, 12)
            img[equipment_zone == 1] = equipment_temp
            hot_x = np.random.randint(70, 154)
            hot_y = np.random.randint(70, 154)
            hot_temp = background_temp + np.random.uniform(25, 45)
            cv2.circle(img, (hot_x, hot_y), 12, hot_temp, -1)
            label = 2
        noise = np.random.normal(0, 2, (img_size, img_size))
        img = img + noise
        gradient = np.linspace(0, 3, img_size).reshape(-1, 1)
        img = img + gradient
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_INFERNO)
        filename = f"real_thermal_{i:04d}.png"
        filepath = os.path.join(images_dir, filename)
        cv2.imwrite(filepath, img_colored)
        grayscale_path = os.path.join(images_dir, f"gray_{i:04d}.png")
        cv2.imwrite(grayscale_path, img_normalized)
        labels[grayscale_path] = label
        if (i + 1) % 50 == 0:
            print(f"   Создано {i+1}/{num_images}...")
    with open("real_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    stats = Counter(labels.values())
    print(f"\n✅ Создан датасет реалистичных термограмм:")
    print(f"   📁 Папка: {images_dir}")
    print(f"   Норма (0): {stats[0]} шт.")
    print(f"   Перегрев (1): {stats[1]} шт.")
    print(f"   Неисправность (2): {stats[2]} шт.")
    return "real_labels.json", images_dir


if __name__ == "__main__":
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ДАННЫХ")
    print("=" * 60)
    print("\n1. Реалистичный датасет (для обучения):")
    create_realistic_training_dataset(num_images=500)
    print("\n2. Физически корректный датасет:")
    generate_realistic_thermal_images(num_images=200)
    print("\n✅ Все данные сгенерированы!")
