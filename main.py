from flask import Flask, request, jsonify
import os
import logging
from typing import Dict

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
ALLOWED_EXTENSIONS = {'.dcm'}


def getPrediction(file: FileStorage) -> Dict[int, float]:
    """
    Метод для получения предсказания по файлу.

    Args:
        file: файл .dcm

    Returns:
        Словарь с ключами 0 и 1 и значениями типа float
    """
    # Здесь должна быть ваша логика обработки DICOM файла
    # Пока что возвращаем mock данные

    # Пример: загрузка и обработка DICOM файла
    try:
        # TODO: Здесь добавить реальную логику обработки файла
        # Будет приходить арихив zip, надо разархивировать, получить предсказание и новый массив
        # Массив превратить в html и вызвать save_file_to_s3(file, filename)
        #
        # Mock предсказание для демонстрации
        import random
        prediction_0 = random.uniform(0.1, 0.6)
        prediction_1 = 1.0 - prediction_0

        return {
            0: prediction_0,
            1: prediction_1
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file.filename}: {e}")
        raise


@app.route('/prediction', methods=['POST'])
def get_prediction():
    """
    Ручка для получения предсказания по файлу.

    URL параметры:
        filename: имя файла для обработки

    Returns:
        JSON с результатом предсказания
    """
    try:
        if 'file' not in request.files:
            return jsonify(error="no file part"), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify(error="empty filename"), 400

        filename = secure_filename(file.filename)

        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return jsonify({
                'error': 'Допустимы только файлы с расширением .dcm'
            }), 400

        # Получаем предсказание
        prediction_dict = getPrediction(file)

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
    # Запускаем сервер
    app.run(host='0.0.0.0', port=8081, debug=True)
