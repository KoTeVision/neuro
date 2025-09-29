import torch
from botocore.exceptions import ClientError
from torch.nn import MSELoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    ToTensord
)
from monai.networks.nets import SwinUNETR  # Импорт SwinUNETR
from monai.inferers import sliding_window_inference
from monai.data import PydicomReader
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # Для интерактивной визуализации

from flask import Flask, jsonify, request
import logging

from werkzeug.exceptions import BadRequest

from save_visuals import save_visuals_to_s3
from utils import *

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
ALLOWED_EXTENSIONS = {'.zip'}

S3_BUCKET_NAME = "kote-uploads"

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

s3 = get_s3_client()

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


def getPrediction(filename: str) -> float:
    """
    Метод для получения предсказания по архиву исследования (zip) в S3.

    Args:
        filename: имя файла в S3 бакете 'kote-uploads' БЕЗ расширения .zip
                  (если передадите 'xxx.zip', тоже отработает)

    Returns:
        (probability, html_content)
        probability: float — вероятность патологии в [0..1]
        html_content: str — HTML с интерактивной визуализацией (такой же, как сохранённый в S3)
    """

    # Локальные временные файлы/папки
    local_zip_path = None
    unzip_path = None

    # Для путей в S3 используем префикс "<filename>/..."
    # Если пользователь передал "... .zip", срежем суффикс для каталога
    s3_prefix = filename[:-4] if filename.endswith(".zip") else filename

    try:
        # 1) Скачиваем zip из S3
        local_zip_path = download_zip_from_s3(s3, S3_BUCKET_NAME, filename)

        # 2) Распаковываем
        unzip_path = unzip_to_temp(local_zip_path)

        # 3) Подготовка данных (используем unzip_path как test_scan_path)
        test_data = [{'image': unzip_path}]
        transformed_data = transforms(test_data[0])
        image = transformed_data['image'].unsqueeze(0).to(device)  # [1, 1, H, W, D]

        # 4) Sliding window inference
        with torch.no_grad():
            reconstructed = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5
            )
            reconstructed = torch.clamp(reconstructed, min=0.0, max=1.0)

        # 5) Ошибка реконструкции
        error = loss_function(reconstructed, image).item()

        # 6) Вероятность патологии (один prediction: float)
        probability_pathology = error_to_probability(error, threshold)

        # 7) Визуализация
        error_map = np.abs(image.cpu().numpy() - reconstructed.cpu().numpy())

        # 8) Сохранить HTML и PNG-срезы в S3 под "<filename>/..."
        save_visuals_to_s3(
            s3=s3,
            bucket=S3_BUCKET_NAME,
            prefix=s3_prefix,
            image_np=image.cpu().numpy(),
            error_np=error_map
        )

        if 'logger' in globals() and logger:
            logger.info(
                f"Готово: filename={filename}, error={error:.6f}, probability={probability_pathology:.6f}, "
                f"S3: s3://{S3_BUCKET_NAME}/{s3_prefix}/visualization.html"
            )

        return probability_pathology

    except Exception as e:
        if 'logger' in globals() and logger:
            logger.error(f"Ошибка при обработке файла {filename}: {e}")
        raise
    finally:
        # Убираем временные файлы
        if local_zip_path:
            try:
                os.remove(local_zip_path)
            except Exception:
                pass
            # Папка, где лежал zip
            try:
                zip_dir = os.path.dirname(local_zip_path)
                safe_rmtree(zip_dir)
            except Exception:
                pass
        if unzip_path:
            safe_rmtree(unzip_path)


@app.route('/prediction', methods=['POST'])
def get_prediction():
    """
    POST /prediction
    Body: { "filename": <string> } — имя zip-файла в бакете S3 (суффикс .zip необязателен).

    Response (200):
      { "result": <float 0..1> }

    Errors:
      400 — некорректный запрос
      404 — файл не найден в S3
      500 — внутренняя ошибка сервиса
    """
    try:
        payload = request.get_json(silent=True) or {}
        filename = payload.get('filename')

        if not isinstance(filename, str) or not filename.strip():
            return jsonify(error="filename is required"), 400
        filename = filename.strip()

        if filename.lower().endswith(".zip"):
            filename = filename[:-4]

        prediction = getPrediction(filename)

        try:
            prob = float(prediction)
        except (TypeError, ValueError):
            logger.error(f"Некорректный формат результата getPrediction: {prediction!r}")
            return jsonify(error="prediction format error"), 500

        if not (0.0 <= prob <= 1.0):
            logger.error(f"Результат вне допустимого диапазона: {prob}")
            return jsonify(error="prediction out of range"), 500

        logger.info(f"Prediction for '{filename}': {prob:.6f}")
        return jsonify(result=prob), 200

    except ClientError as ce:
        code = ce.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            logger.warning(f"S3 object not found for filename='{filename}': {code}")
            return jsonify(error="file not found in S3"), 404
        logger.error(f"S3 client error for '{filename}': {ce}")
        return jsonify(error="S3 error"), 500

    except BadRequest as br:
        logger.error(f"Bad request: {br}")
        return jsonify(error="bad request"), 400

    except Exception as e:
        logger.error(f"Internal error while processing '{filename}': {e}")
        return jsonify(error="internal server error"), 500


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
