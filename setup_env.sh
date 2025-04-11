#!/bin/bash

# Обновленный скрипт для инициализации окружения Python для запуска
# примера кода NoProp с использованием файла requirements.txt.

# --- Конфигурация ---
ENV_NAME="noprop_env"          # Имя виртуального окружения
PYTHON_CMD="python3"          # Используйте python3 или python
REQUIREMENTS_FILE="requirements.txt" # Имя файла с зависимостями
SCRIPT_NAME="noprop_example.py"  # Имя вашего Python скрипта

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
echo "Файл зависимостей: ${REQUIREMENTS_FILE}"
echo "---------------------------------"

# === Использование venv (стандартный модуль Python) ===
echo ""
echo "Использую 'venv'..."

check_command ${PYTHON_CMD}
check_command pip

# Проверка наличия файла requirements.txt
if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "Ошибка: Файл зависимостей '${REQUIREMENTS_FILE}' не найден в текущей директории."
    echo "Пожалуйста, создайте его со списком пакетов."
    exit 1
fi

echo "1. Создание виртуального окружения '${ENV_NAME}' (если не существует)..."
if [ ! -d "${ENV_NAME}" ]; then
  ${PYTHON_CMD} -m venv ${ENV_NAME}
  if [ $? -ne 0 ]; then
    echo "Ошибка: Не удалось создать виртуальное окружение с помощью venv."
    exit 1
  fi
else
  echo "   Виртуальное окружение '${ENV_NAME}' уже существует."
fi


echo "2. Активация окружения (для этого скрипта)..."
# Обратите внимание: эта активация действует только внутри скрипта.
source ${ENV_NAME}/bin/activate

echo "3. Обновление pip..."
pip install --upgrade pip

echo "4. Установка зависимостей из '${REQUIREMENTS_FILE}'..."
pip install -r ${REQUIREMENTS_FILE}

if [ $? -ne 0 ]; then
    echo "Ошибка: Не удалось установить зависимости из '${REQUIREMENTS_FILE}'."
    deactivate &> /dev/null # Попытка деактивировать
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


# === Вариант Conda (закомментирован) ===
# Если вы предпочитаете Conda, раскомментируйте этот блок
# и закомментируйте блок venv выше. Убедитесь, что Conda установлена.
# echo ""
# echo "Пытаюсь использовать 'conda'..."
# check_command conda
# if [ ! -f "${REQUIREMENTS_FILE}" ]; then
#     echo "Ошибка: Файл зависимостей '${REQUIREMENTS_FILE}' не найден."
#     exit 1
# fi
# echo "1. Создание/Обновление Conda окружения '${ENV_NAME}'..."
# # Проверяем, существует ли окружение
# if conda info --envs | grep -q "${ENV_NAME}"; then
#   echo "   Окружение Conda '${ENV_NAME}' уже существует. Установка/Обновление пакетов..."
#   conda activate ${ENV_NAME}
#   # Conda может не поддерживать все версии из pip, используем pip внутри conda
#   pip install -r ${REQUIREMENTS_FILE}
# else
#   echo "   Создание нового окружения Conda '${ENV_NAME}'..."
#   # Укажите нужную версию python
#   conda create --name ${ENV_NAME} python=3.10 -y # Например, python 3.10
#   if [ $? -ne 0 ]; then echo "Ошибка создания окружения Conda."; exit 1; fi
#   eval "$(conda shell.bash hook)" # Для активации в скрипте
#   conda activate ${ENV_NAME}
#   echo "   Установка зависимостей из '${REQUIREMENTS_FILE}' с помощью pip..."
#   pip install --upgrade pip # Обновим pip внутри conda
#   pip install -r ${REQUIREMENTS_FILE}
# fi
# if [ $? -ne 0 ]; then
#     echo "Ошибка: Не удалось установить зависимости в окружении Conda."
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
echo "Не забудьте активировать окружение '${ENV_NAME}' ('source ${ENV_NAME}/bin/activate') перед запуском Python кода!"