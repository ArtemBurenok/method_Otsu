import unittest
import numpy as np
from Otsu import image_processing


class TestOtsu(unittest.TestCase):
    def setUp(self):
        self.image1 = 'images/jupiter.jpg'  # изображение, которое не имеет одного явно выраженного объекта
        self.image2 = 'images/big_brother.png'
        self.noise_image = 'images/noise_image.png'  # изображение с шумом
        self.empty_image = 'images/empty_image.png'  # пустое изображение
        self.same_value_image = 'images/same_value_image.png'  # изображение с одним цветом
        self.gradient_image = 'images/gradient_image.png'  # изображение с градиентом

    # тест на обычных изображениях (одно из них не имеет одного явно выраженного объекта)
    def test_otsu_threshold(self):
        # применение метода Оцу к изображениям
        thresholded_image1, _ = image_processing(self.image1).otsu()
        thresholded_image2, _ = image_processing(self.image2).otsu()

        self.assertTrue(np.array_equal(np.unique(thresholded_image1), [0, 255]))
        self.assertTrue(np.array_equal(np.unique(thresholded_image2), [0, 255]))

    # тест на то, что выходное изображение имеет ту же форму, что и входное
    def test_otsu_shape(self):
        # применение метода Оцу к изображениям
        thresholded_image1, _ = image_processing(self.image1).otsu()
        thresholded_image2, _ = image_processing(self.image2).otsu()

        # получение размерностей изображений
        size_image1 = image_processing(self.image1).image.shape
        size_image2 = image_processing(self.image2).image.shape

        self.assertEqual(thresholded_image1.shape, (size_image1[0], size_image1[1]))
        self.assertEqual(thresholded_image2.shape, (size_image2[0], size_image2[1]))

    # тест на изображении c шумом
    def test_otsu_noisy(self):
        # применение метода Оцу к изображению
        _, threshold = image_processing(self.noise_image).otsu()

        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 255)

    # тест на пустом изображении
    def test_otsu_noisy(self):
        # применение метода Оцу к изображению
        _, threshold = image_processing(self.empty_image).otsu()

        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 255)

    # тест на изображении с одинаковыми значениями
    def test_otsu_same_image(self):
        # применение метода Оцу к изображению
        thresholded_image, _ = image_processing(self.same_value_image).otsu()

        self.assertTrue(np.array_equal(np.unique(thresholded_image), [255]))

    # тест на изображении с градиентом
    def test_otsu_gradient(self):
        # применение метода Оцу к изображению
        thresholded_image, _ = image_processing(self.gradient_image).otsu()

        # полученный результат должен иметь два цвета
        unique_values = np.unique(thresholded_image)
        self.assertEqual(len(unique_values), 2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
