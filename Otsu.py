import numpy as np
from PIL import Image  # для загрузки изображения
import matplotlib.pyplot as plt


class image_processing:
    def __init__(self, image):
        self._image = np.array(Image.open(image))

    # геттер переменной image
    @property
    def image(self):
        return self._image

    # сеттер переменной image
    @image.setter
    def image(self, new_image):
        self._image = new_image

    # преобразование с помощью взвешенных коэффициентов цветного изображения в градации серого
    def grayscale(self):
        grayscale = 0.29 * self._image[:, :, 0] + 0.58 * self._image[:, :, 1] + 0.11 * self._image[:, :, 2]

        return grayscale

    # преобразование изображения в черно-белое по заданному порогу
    def binarize(self, threshold=128):
        grayscale = self.grayscale()
        binary_image = np.where(grayscale > threshold, 255, 0)

        return binary_image

    # сохранение изображения
    def save_image(self, image_array, output_path):
        image = Image.fromarray(image_array.astype(np.uint8))
        image.save(output_path)

    # отображение изображения
    def plot_image(self, image):
        plt.imshow(image)
        plt.show()

    # реализация метода Оцу
    def otsu(self):
        # преобразование изображения в градации серого
        gray_image = self.grayscale()

        # построение гистограммы значений яркости
        histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0, 255))

        # общее количество пикселей в изображении
        total_pixels = gray_image.size

        # начальные значения
        current_max = 0  # максимальное значение межклассовой дисперсии
        threshold = 0  # оптимальный порог
        sum_total = np.dot(np.arange(256), histogram)  # общая сумма всех значений яркости
        sum_background = 0  # сумма значений яркости для фона
        weight_background = 0  # количество пикселей, относящихся к фону
        weight_foreground = 0  # количество пикселей, не относящихся к фону

        for i in range(256):
            weight_background += histogram[i]

            if weight_background == 0:
                continue

            # разность между общим количеством пикселей и весом фона
            weight_foreground = total_pixels - weight_background

            if weight_foreground == 0:
                break

            # обновление суммы яркости для фона и расчёт суммы яркостей для пикселей, которые не принадлежат фону
            sum_background += i * histogram[i]
            sum_foreground = sum_total - sum_background

            # находятся средние значения для фона и объекта
            mean_background = sum_background / weight_background
            mean_foreground = sum_foreground / weight_foreground

            # расчёт межклассовой дисперсии
            between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

            # нахождение максимального значения дисперсии
            if between_class_variance > current_max:
                current_max = between_class_variance
                threshold = i

        # бинаризация
        new_image = self.binarize(threshold)

        return new_image, threshold
