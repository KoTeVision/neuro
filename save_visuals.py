import io
from typing import List
from matplotlib import pyplot as plt
from utils import *


def save_visuals_to_s3(
    s3,
    bucket: str,
    prefix: str,
    image_np: np.ndarray, # [1, 1, H, W, D]
    error_np: np.ndarray  # [1, 1, H, W, D]
) -> List[str]:
    """
    Сохраняет для КАЖДОГО среза по оси D один PNG-оверлей:
    - подложка: оригинальный срез (градации серого),
    - поверх: тепловая карта ошибки (полупрозрачная).

    Файлы кладутся в S3 под ключами:
      {prefix}/1.png, {prefix}/2.png, ..., {prefix}/D.png

    Возвращает:
      List[str] — список сохранённых ключей в порядке возрастания (1..D).

    Требования:
      - image_np и error_np имеют форму [1,1,H,W,D].
      - Функция upload_bytes(s3, bucket, key, data, content_type) доступна.
    """
    # ---- валидация формы ----
    for name, arr in (("image_np", image_np), ("error_np", error_np)):
        if not (isinstance(arr, np.ndarray) and arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1):
            raise ValueError(f"{name}: ожидается тензор формы [1,1,H,W,D], получено {getattr(arr, 'shape', None)}")

    H, W, D = image_np.shape[2], image_np.shape[3], image_np.shape[4]
    if D <= 0:
        raise ValueError("Пустой объём: D == 0")

    base_vol = image_np[0, 0]   # [H, W, D]
    err_vol  = error_np[0, 0]   # [H, W, D]

    # ---- глобальная нормализация по объёму (чтобы яркость не 'прыгала' между срезами) ----
    def _robust_minmax(vol: np.ndarray, low=1, high=99):
        finite = vol[np.isfinite(vol)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin, vmax = np.percentile(finite, [low, high])
        if not np.isfinite(vmin): vmin = float(np.nanmin(finite))
        if not np.isfinite(vmax): vmax = float(np.nanmax(finite))
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    base_vmin, base_vmax = _robust_minmax(base_vol, 1, 99)
    err_vmin,  err_vmax  = _robust_minmax(err_vol,  1, 99)

    saved_keys: List[str] = []

    # ---- генерация и загрузка PNG для каждого среза ----
    for i in range(D):
        base_slice = base_vol[:, :, i]
        err_slice  = err_vol[:,  :, i]

        # нормализация ошибки в [0,1] для корректной теплокарты
        if err_vmax - err_vmin < 1e-9:
            err_norm = np.zeros_like(err_slice, dtype=float)
        else:
            err_norm = (err_slice - err_vmin) / (err_vmax - err_vmin)
            err_norm = np.clip(err_norm, 0.0, 1.0)
            err_norm[~np.isfinite(err_norm)] = 0.0

        # рисуем оверлей
        buf = io.BytesIO()
        plt.figure()
        plt.imshow(base_slice, cmap="gray", vmin=base_vmin, vmax=base_vmax)
        plt.imshow(err_norm, cmap="magma", alpha=0.55)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.0)
        plt.close()
        buf.seek(0)

        key = f"{prefix}/{i+1}.png"
        try:
            upload_bytes(
                s3,
                bucket,
                key=key,
                data=buf.read(),
                content_type="image/png",
            )
        except Exception as e:

            raise

        saved_keys.append(key)

    return saved_keys
