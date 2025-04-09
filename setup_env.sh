#!/bin/bash

# Скрипт для инициализации окружения Python для запуска
# примера кода NoProp с использованием PyTorch.

# --- Конфигурация ---
ENV_NAME="noprop_env"  # Имя виртуального окружения
PYTHON_CMD="python3"   # Используйте python3 или python, в зависимости от вашей системы
SCRIPT_NAME="noprop_example.py" # Имя вашего Python скрипта с примером NoProp

# --- Функции ---
check_command() {
  if ! command -v $1 &> /dev/null
  then
    echo "Ошибка: Команда '$1' не найдена. Пожалуйста, установите $1 и попробуйте снова."
    exit 1
  fi
}

echo "Настройка окружения для NoProp..."
echo "Имя окружения: ${ENV_NAME}"
echo "---------------------------------"

# === Вариант 1: Использование venv (стандартный модуль Python) ===
echo ""
echo "Пытаюсь использовать 'venv'..."

check_command ${PYTHON_CMD}
check_command pip

echo "1. Создание виртуального окружения '${ENV_NAME}'..."
${PYTHON_CMD} -m venv ${ENV_NAME}

if [ $? -ne 0 ]; then
    echo "Ошибка: Не удалось создать виртуальное окружение с помощью venv."
    exit 1
fi

echo "2. Активация окружения (для этого скрипта)..."
# Обратите внимание: эта активация действует только внутри скрипта.
# Вам нужно будет активировать окружение вручную в вашем терминале.
source ${ENV_NAME}/bin/activate

echo "3. Обновление pip..."
pip install --upgrade pip

echo "4. Установка необходимых библиотек (PyTorch, Torchvision, NumPy)..."
# **ВАЖНО**: Команда установки PyTorch может отличаться в зависимости от
# вашей ОС и наличия CUDA (GPU).
# Пожалуйста, проверьте официальный сайт PyTorch для получения правильной команды:
# https://pytorch.org/get-started/locally/
# Пример для CPU-only:
pip install torch torchvision numpy
# Пример для CUDA 11.8 (может измениться):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Пример для CUDA 12.1 (может измениться):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


if [ $? -ne 0 ]; then
    echo "Ошибка: Не удалось установить зависимости с помощью pip."
    # Попытка деактивировать, если активация произошла
    deactivate &> /dev/null
    exit 1
fi

echo ""
echo "--- Установка с помощью venv завершена ---"
echo "Чтобы активировать окружение в вашем терминале, выполните:"
echo "source ${ENV_NAME}/bin/activate"
echo ""
echo "Чтобы запустить Python скрипт (${SCRIPT_NAME}), после активации выполните:"
echo "python ${SCRIPT_NAME}"
echo ""
echo "Чтобы деактивировать окружение, выполните:"
echo "deactivate"
echo "---------------------------------"

# === Вариант 2: Использование Conda (если вы предпочитаете Anaconda/Miniconda) ===
# Закомментируйте или удалите раздел venv выше, если хотите использовать только Conda.
# ИЛИ раскомментируйте этот раздел, если установлен Conda.

# echo ""
# echo "Пытаюсь использовать 'conda'..."

# check_command conda

# echo "1. Создание Conda окружения '${ENV_NAME}'..."
# # Вы можете указать конкретную версию Python, например python=3.9
# conda create --name ${ENV_NAME} python=3.9 -y

# if [ $? -ne 0 ]; then
#     echo "Ошибка: Не удалось создать окружение Conda."
#     exit 1
# fi

# echo "2. Активация окружения (для этого скрипта)..."
# # Обратите внимание: активация Conda может потребовать инициализации оболочки (`conda init`)
# # Эта активация действует только внутри скрипта.
# eval \"$(conda shell.bash hook)\" # Необходимо для активации в скрипте
# conda activate ${ENV_NAME}

# echo "3. Установка необходимых библиотек (PyTorch, Torchvision, NumPy) через Conda..."
# # **ВАЖНО**: Команда установки PyTorch может отличаться.
# # Используйте официальный сайт PyTorch для выбора правильной команды Conda:
# # https://pytorch.org/get-started/locally/
# # Пример для CPU-only:
# conda install pytorch torchvision cpuonly -c pytorch -y
# # Пример для CUDA 11.8 (может измениться):
# # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# # Пример для CUDA 12.1 (может измениться):
# # conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# # Установка NumPy
# conda install numpy -y

# if [ $? -ne 0 ]; then
#     echo "Ошибка: Не удалось установить зависимости с помощью Conda."
#     conda deactivate &> /dev/null
#     exit 1
# fi

# echo ""
# echo "--- Установка с помощью Conda завершена ---"
# echo "Чтобы активировать окружение в вашем терминале, выполните:"
# echo "conda activate ${ENV_NAME}"
# echo ""
# echo "Чтобы запустить Python скрипт (${SCRIPT_NAME}), после активации выполните:"
# echo "python ${SCRIPT_NAME}"
# echo ""
# echo "Чтобы деактивировать окружение, выполните:"
# echo "conda deactivate"
# echo "---------------------------------"


echo ""
echo "Скрипт инициализации завершен."
echo "Не забудьте активировать окружение '${ENV_NAME}' перед запуском Python кода!"