import subprocess
import sys
import os


def run_script(script_name, args=None):
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    print(f"\n🚀 Запуск: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Ошибка при выполнении {script_name}")
        sys.exit(result.returncode)
    return result


def main():
    print("=" * 60)
    print("🚀 ЗАПУСК ВСЕГО ПАЙПЛАЙНА")
    print("=" * 60)
    print("\n📊 Шаг 1: Генерация данных")
    run_script("src/generate_data.py")
    print("\n🧠 Шаг 2: Обучение модели")
    run_script("src/train.py")
    print("\n📸 Шаг 3: Демонстрация анализа")
    if not os.path.exists("thermal_images"):
        os.makedirs("thermal_images")
        print("📁 Создана папка thermal_images/ для тестовых изображений")
        print("📌 Поместите термограммы в папку и запустите:")
        print("   python src/analyze_folder.py --folder thermal_images")
    else:
        run_script("src/analyze_folder.py", ["--folder", "thermal_images"])
    print("\n" + "=" * 60)
    print("✅ ПАЙПЛАЙН ЗАВЕРШЁН!")
    print("=" * 60)


if __name__ == "__main__":
    main()
