import os
import argparse

import torch
import cv2
import numpy as np

from utils import ImprovedThermalNet, load_image, get_weather, get_default_weather, CLASSES, ICONS


def predict_single(image_path, model_path="models/best_model.pth", weather_api_key=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedThermalNet(num_classes=3).to(device)
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("Сначала запустите: python src/train.py")
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    img = load_image(image_path)
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).item()
    weather = get_weather(api_key=weather_api_key)
    if weather is None:
        weather = get_default_weather()
    return {
        'file': os.path.basename(image_path),
        'diagnosis': CLASSES[pred],
        'class_id': pred,
        'icon': ICONS[pred],
        'probabilities': {
            'norm': float(probs[0][0]),
            'overheat': float(probs[0][1]),
            'fault': float(probs[0][2])
        },
        'weather': weather
    }


def print_result(result):
    if result is None:
        return
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ ДИАГНОСТИКИ")
    print("=" * 60)
    print(f"\n📸 {result['file']}")
    print(f"   {result['icon']} Диагноз: {result['diagnosis']}")
    print(f"   📊 Вероятности:")
    print(f"      Норма: {result['probabilities']['norm']*100:.1f}%")
    print(f"      Перегрев: {result['probabilities']['overheat']*100:.1f}%")
    print(f"      Неисправность: {result['probabilities']['fault']*100:.1f}%")
    if result['class_id'] == 0:
        print(f"   💡 Оборудование в норме")
    elif result['class_id'] == 1:
        if result['weather']['temperature'] > 25:
            print(f"   💡 Перегрев! Высокая внешняя температура усугубляет ситуацию")
        else:
            print(f"   💡 ВНИМАНИЕ! Обнаружен перегрев! Проверьте охлаждение")
    else:
        print(f"   💡 ОПАСНО! Требуется немедленный ремонт!")
    print(f"\n🌤️ Погода: {result['weather']['city']}")
    print(f"   🌡️ Температура: {result['weather']['temperature']:.1f}°C")
    print(f"   💧 Влажность: {result['weather']['humidity']}%")
    print(f"   ☁️ {result['weather']['description']}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Диагностика термограммы")
    parser.add_argument('--image', type=str, required=True, help="Путь к изображению")
    parser.add_argument('--model', type=str, default="models/best_model.pth", help="Путь к модели")
    parser.add_argument('--api_key', type=str, help="API ключ OpenWeatherMap")
    args = parser.parse_args()
    if not os.path.exists(args.image):
        print(f"❌ Изображение не найдено: {args.image}")
        return
    result = predict_single(args.image, args.model, args.api_key)
    print_result(result)


if __name__ == "__main__":
    main()
