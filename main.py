from flask import Flask, request, jsonify
import os
import logging
from typing import Dict, Union

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
UPLOAD_DIR = "./uploads"  # Директория где хранятся файлы
ALLOWED_EXTENSIONS = {'.dicom'}


def getPrediction(file_path: str) -> Dict[int, float]:
    """
    Метод для получения предсказания по файлу.

    Args:
        file_path: Путь к файлу

    Returns:
        Словарь с ключами 0 и 1 и значениями типа float
    """
    # Здесь должна быть ваша логика обработки DICOM файла
    # Пока что возвращаем mock данные

    # Пример: загрузка и обработка DICOM файла
    try:
        # TODO: Здесь добавить реальную логику обработки файла
        # Например:
        # import pydicom
        # dicom_data = pydicom.dcmread(file_path)
        # processed_data = preprocess_dicom(dicom_data)
        # prediction = model.predict(processed_data)

        # Mock предсказание для демонстрации
        import random
        prediction_0 = random.uniform(0.1, 0.6)
        prediction_1 = 1.0 - prediction_0

        return {
            0: prediction_0,
            1: prediction_1
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        raise


@app.route('/prediction', methods=['GET'])
def get_prediction():
    """
    Ручка для получения предсказания по файлу.

    URL параметры:
        filename: имя файла для обработки

    Returns:
        JSON с результатом предсказания
    """
    try:
        # Получаем имя файла из URL параметров
        filename = request.args.get('filename')

        if not filename:
            return jsonify({
                'error': 'Параметр filename обязателен'
            }), 400

        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return jsonify({
                'error': 'Допустимы только файлы с расширением .dicom'
            }), 400

        # Формируем полный путь к файлу
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Проверяем существование файла
        if not os.path.exists(file_path):
            return jsonify({
                'error': f'Файл {filename} не найден'
            }), 404

        # Проверяем что это файл, а не директория
        if not os.path.isfile(file_path):
            return jsonify({
                'error': f'{filename} не является файлом'
            }), 400

        # Получаем предсказание
        prediction_dict = getPrediction(file_path)

        # Проверяем формат ответа
        if not isinstance(prediction_dict, dict) or 1 not in prediction_dict:
            logger.error(f"Неверный формат ответа от getPrediction: {prediction_dict}")
            return jsonify({
                'error': 'Ошибка обработки файла'
            }), 500

        # Возвращаем результат
        result = prediction_dict[1]

        logger.info(f"Обработан файл {filename}, результат: {result}")

        return jsonify({
            'result': result
        })

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        return jsonify({
            'error': 'Внутренняя ошибка сервера'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'prediction-service'
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint не найден'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500


if __name__ == '__main__':
    # Создаем директорию для файлов, если она не существует
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Запускаем сервер
    app.run(host='0.0.0.0', port=8081, debug=True)