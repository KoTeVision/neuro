# Базовый образ с Python 3.11
FROM python:3.11-slim

# Отключаем запись .pyc и включаем буферизацию stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Рабочая директория
WORKDIR /app

# Устанавливаем системные зависимости для сборки Python пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаём папку для загрузок файлов
RUN mkdir -p /app/uploads

# Открываем порт
EXPOSE 8081

# Запускаем сервис
CMD ["python", "main.py"]
