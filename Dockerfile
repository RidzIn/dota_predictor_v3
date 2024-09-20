# Используем базовый образ Python
FROM python:3.11.5

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы в контейнер
COPY . /app

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Обновляем scikit-learn до версии 1.5.1
RUN pip install scikit-learn==1.5.1 --force-reinstall --no-deps

# Указываем порт
EXPOSE 8501

# Команда для запуска приложения
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

