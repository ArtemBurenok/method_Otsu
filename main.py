from Otsu import image_processing

if __name__ == "__main__":
    paths_list = ['images/jupiter.jpg', 'images/big_brother.png', 'images/noise_image.png',
                  'images/empty_image.png', 'images/same_value_image.png', 'images/gradient_image.png']

    for path in paths_list:
        processing = image_processing(path)

        # бинаризация
        bin_image = processing.binarize(128)
        processing.save_image(bin_image, f'changed_images/binarized/bin_{path.split("/")[1]}')

        # градиент серого
        grayscale_image = processing.grayscale()
        processing.save_image(grayscale_image, f'changed_images/grayscaled/gray_{path.split("/")[1]}')

        # метод Оцу
        otsu_image, _ = processing.otsu()
        processing.save_image(otsu_image, f'changed_images/Otsu_processed/Otsu_{path.split("/")[1]}')
