import os
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse

from predict import predict_single
from utils import get_weather, get_default_weather
from utils import CLASSES, ICONS


def analyze_folder(folder_path, model_path="models/best_model.pth", weather_api_key=None):
    if not os.path.exists(folder_path):
        print(f"❌ Папка не найдена: {folder_path}")
        return
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        images.extend(Path(folder_path).glob(ext))
    if len(images) == 0:
        print(f"❌ В папке '{folder_path}' нет изображений")
        return
    print(f"\n📸 Найдено {len(images)} термограмм в папке '{folder_path}'")
    weather = get_weather(api_key=weather_api_key)
    if weather is None:
        weather = get_default_weather()
    print(f"\n🌤️ Погода: {weather['city']}")
    print(f"   🌡️ Температура: {weather['temperature']:.1f}°C")
    results = []
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
    print("=" * 60)
    for i, img_path in enumerate(images, 1):
        result = predict_single(str(img_path), model_path, weather_api_key)
        if result:
            results.append(result)
            print(f"\n📸 {i}/{len(images)}: {img_path.name}")
            print(f"   {ICONS[result['class_id']]} Диагноз: {result['diagnosis']}")
            print(f"   📊 Вероятности:")
            print(f"      Норма: {result['probabilities']['norm']*100:.1f}%")
            print(f"      Перегрев: {result['probabilities']['overheat']*100:.1f}%")
            print(f"      Неисправность: {result['probabilities']['fault']*100:.1f}%")
    report = {
        'timestamp': datetime.now().isoformat(),
        'weather': weather,
        'total_images': len(images),
        'results': results
    }
    with open('diagnosis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ ОТЧЁТА")
    print("=" * 60)
    print(f"✅ Отчёт сохранён в 'diagnosis_report.json'")
    diagnoses = [r['diagnosis'] for r in results]
    stats = Counter(diagnoses)
    print("\n📊 ОБЩАЯ СТАТИСТИКА:")
    for diag, count in stats.items():
        print(f"   {diag}: {count} шт. ({count/len(results)*100:.1f}%)")
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Диагностика всех термограмм в папке")
    parser.add_argument('--folder', type=str, default="thermal_images", help="Папка с изображениями")
    parser.add_argument('--model', type=str, default="models/best_model.pth", help="Путь к модели")
    parser.add_argument('--api_key', type=str, help="API ключ OpenWeatherMap")
    args = parser.parse_args()
    analyze_folder(args.folder, args.model, args.api_key)


if __name__ == "__main__":
    main()
