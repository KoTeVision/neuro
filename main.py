import torch  # TODO: занести импорты в requirements.txt
from torch.nn import MSELoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    ToTensord
)
from monai.networks.nets import SwinUNETR  # Импорт SwinUNETR
from monai.inferers import sliding_window_inference
from monai.data import PydicomReader
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Для интерактивной визуализации
import plotly.io as pio  # Для экспорта или отображения

from flask import Flask, request, jsonify
import os
import logging
from typing import Dict, Tuple

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
ALLOWED_EXTENSIONS = {'.zip'}

# Временная папка для разархивирования ZIP
TEMP_DIR = 'temp_unzip'
os.makedirs(TEMP_DIR, exist_ok=True)

# Путь к модели
model_path = 'swin_unetr_norm_96.pt'

# Загрузка модели (загружается один раз при запуске сервера)
model = SwinUNETR(
    in_channels=1,  # Grayscale CT
    out_channels=1,  # Реконструкция
    feature_size=48,  # Базовый размер фич
    use_checkpoint=True  # Для экономии памяти
)

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка весов модели (с фильтрацией совместимых слоёв)
model_dict = model.state_dict()  # Текущий state_dict модели
pretrained_dict = torch.load(
    model_path,
    map_location=torch.device('cpu'),
    weights_only=True
)  # Загружаем обученный state_dict

# Фильтруем только совместимые ключи (по имени и форме)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

# Обновляем модель совместимыми весами
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)  # Загружаем отфильтрованный state_dict
print("Веса модели загружены с фильтрацией совместимых слоёв.")
model.to(device)
model.eval()

# Loss для вычисления ошибки
loss_function = MSELoss()

# Порог для аномалии (threshold): Рассчитай на val-норме (mean MSE + 2*std)
threshold = 0.02  # Пример; подставь свой расчёт

# Преобразования для теста (без аугментации и ресайза)
transforms = Compose([
    LoadImaged(keys=['image'], reader=PydicomReader(sort_fn=None)),  # Чтение DICOM как 3D volume
    EnsureChannelFirstd(keys=['image']),  # Добавляем канал
    ScaleIntensityRanged(keys=['image'], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    # Нормализация HU для КТ
    ToTensord(keys=['image'])
])


def find_dicom_dir(unzip_path: str) -> str:
    """
    Рекурсивно находит папку с .dcm файлами внутри unzip_path.
    """
    for root, dirs, files in os.walk(unzip_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if dcm_files:
            return root  # Возвращаем папку с .dcm
    raise ValueError("Не найдено .dcm файлов в ZIP-архиве.")


def unzip_file(zip_file: FileStorage) -> str:
    """
    Разархивирует ZIP-файл во временную папку.

    Args:
        zip_file: ZIP-файл

    Returns:
        Путь к разархивированной папке с DICOM
    """
    import zipfile
    zip_filename = secure_filename(zip_file.filename)
    zip_path = os.path.join(TEMP_DIR, zip_filename)
    zip_file.save(zip_path)

    unzip_path = os.path.join(TEMP_DIR, os.path.splitext(zip_filename)[0])
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)

    # Удаляем ZIP после разархивирования
    os.remove(zip_path)

    return unzip_path  # Путь к корню разархивированного ZIP


# Функция для наложения маски на срез с прозрачностью
def overlay_mask_on_slice(slice_img, slice_mask, alpha=0.5):
    """
    Накладывает маску (error_map) на срез изображения в красных оттенках с прозрачностью.
    - slice_img: 2D-срез оригинала (grayscale, [0,1])
    - slice_mask: 2D-срез маски ошибок (нормализован в [0,1])
    - alpha: Прозрачность маски
    """
    # Обрезаем лишние измерения, если есть
    slice_img = np.squeeze(slice_img)
    slice_mask = np.squeeze(slice_mask)

    # Нормализуем маску
    slice_mask = (slice_mask - np.min(slice_mask)) / (np.max(slice_mask) - np.min(slice_mask) + 1e-8)
    slice_mask = np.clip(slice_mask, 0, 1)  # Убеждаемся, что маска в [0, 1]

    # Создаём RGB-изображение для оригинала
    img_rgb = np.stack([slice_img] * 3, axis=-1)  # [H, W, 3]

    # Создаём маску в красных оттенках
    mask_rgb = plt.cm.Reds(slice_mask)[:, :, :3]  # [H, W, 3]

    # Накладываем маску с прозрачностью и обрезаем до [0, 1]
    overlaid = img_rgb * (1 - alpha * slice_mask[..., np.newaxis]) + mask_rgb * (alpha * slice_mask[..., np.newaxis])
    overlaid = np.clip(overlaid, 0, 1)  # Обрезаем финальный результат до [0, 1]

    return overlaid


def generate_plotly_fig(original, reconstructed, error_map, threshold=0.02):
    """
    Генерирует Plotly фигуру с слайдером для просмотра срезов (полная версия, как в test_SwinUnetr.py).
    - original: 3D NumPy [B, C, H, W, D]
    - reconstructed: 3D NumPy [B, C, H, W, D]
    - error_map: 3D NumPy [B, C, H, W, D]
    - threshold: Порог для патологии
    """
    # Вычисляем глобальный MSE для классификации
    global_mse = np.mean(np.abs(original - reconstructed))
    prediction = "патология (аномалия)" if global_mse > threshold else "норма"
    print(f"Глобальный MSE: {global_mse:.4f}, Предсказание: {prediction}")

    # Обрезаем лишние измерения
    original = np.squeeze(original)
    reconstructed = np.squeeze(reconstructed)
    error_map = np.squeeze(error_map)

    # Количество срезов (по глубине Z)
    num_slices = original.shape[2]
    effective_num = num_slices  # Без шага, как в твоей версии (можно добавить step=2 если много)

    # Создаём фигуру
    fig = go.Figure()

    # Добавляем traces для каждого среза (невидимые, кроме первого)
    for idx in range(effective_num):
        slice_img = original[:, :, idx]
        slice_error = error_map[:, :, idx]
        # Накладываем маску, если патология
        if prediction == "патология (аномалия)":
            overlaid = overlay_mask_on_slice(slice_img, slice_error,
                                             alpha=0.5) * 255  # Умножаем на 255 для видимого RGB
            fig.add_trace(go.Image(z=overlaid.astype(np.uint8), colormodel='rgb', visible=(idx == effective_num // 2)))
        else:
            fig.add_trace(go.Image(z=slice_img * 255, colorscale='gray', visible=(idx == effective_num // 2)))

    # Слайдер для переключения видимости
    sliders = [dict(
        active=effective_num // 2,
        yanchor="top",
        xanchor="center",
        currentvalue={"prefix": "Срез: ", "xanchor": "center"},
        pad={"t": 50},
        len=0.9,
        x=0.05,
        y=0,
        steps=[dict(
            method="update",
            args=[{"visible": [k == j for k in range(effective_num)]},
                  {"title": f"Срез {j + 1}/{num_slices} | Предсказание: {prediction}"}],
            label=str(j + 1)
        ) for j in range(effective_num)]
    )]

    fig.update_layout(
        title=f"Срез {effective_num // 2 + 1}/{num_slices} | Предсказание: {prediction}",
        width=800, height=800,  # Увеличенный размер для детализации
        margin=dict(l=0, r=0, t=50, b=0),  # Минимальные отступы для центрирования
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',  # Прозрачный фон
        sliders=sliders
    )

    return fig  # Возвращаем фигуру для сохранения в HTML


def getPrediction(file: FileStorage) -> Tuple[Dict[int, float], str]:
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
        # Разархивировать ZIP
        unzip_path = unzip_file(file)

        # Найти папку с .dcm
        dicom_dir = find_dicom_dir(unzip_path)

        # Подготовка данных (используем unzip_path как test_scan_path)
        test_data = [{'image': dicom_dir}]
        transformed_data = transforms(test_data[0])
        image = transformed_data['image'].unsqueeze(0).to(device)  # [1, 1, H, W, D]

        # Sliding window inference
        with torch.no_grad():
            reconstructed = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 96),  # Размер патча
                sw_batch_size=4,
                predictor=model,
                overlap=0.5
            )
            reconstructed = torch.clamp(reconstructed, min=0.0, max=1.0)

        # Вычисление MSE
        error = loss_function(reconstructed, image).item()

        # Классификация
        prediction_0 = 1.0 if error > threshold else 0.0  # TODO: возвращать вероятность нормы и патлологии, а не результат 0 или 1
        prediction_1 = 1.0 - prediction_0  # 0 - норма, 1 - патология

        # Генерация HTML с визуализацией, если патология
        error_map = np.abs(image.cpu().numpy() - reconstructed.cpu().numpy())
        fig = generate_plotly_fig(image.cpu().numpy(), reconstructed.cpu().numpy(), error_map,
                                  threshold)  # Твоя функция interactive_visualize, но возвращает fig

        # Или так (сохраняет html в память)
        # html_path = os.path.join('static', 'visualization.html')  # Сохраняем в static для доступа
        # pio.write_html(fig, html_path)

        # Или так (не сохраняется с тупа строкой в формате html передает дальше) должно быть шустрее
        html_content = pio.to_html(fig, full_html=True)  # Полный HTML как строка
        # s3_url = save_to_s3(html_content, 'your-bucket-name', 'visualization.html')  # Замени 'your-bucket-name'
        # html_path = s3_url if s3_url else None

        # Очистка временной папки
        for root, dirs, files in os.walk(unzip_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(unzip_path)

        return {
            0: prediction_0,
            1: prediction_1,
        }, html_content

        # TODO: Здесь добавить реальную логику обработки файла
        # Будет приходить арихив zip, надо разархивировать, получить предсказание и новый массив
        # Массив превратить в html и вызвать save_file_to_s3(file, filename)

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
                'error': 'Допустимы только файлы с расширением .zip'
            }), 400

        # Получаем предсказание
        prediction_dict, html_content = getPrediction(file)

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
            'result': result,
            'html_content': html_content
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
