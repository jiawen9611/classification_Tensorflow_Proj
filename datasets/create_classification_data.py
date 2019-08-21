# -*- coding: utf-8 -*-

# coder: Jiawen Zhu
# date: 2019.6.9
# state: modified

import cv2
import numpy as np
from captcha.image import ImageCaptcha

'''
创建简单的captcha数据集用于分类小实验
'''


def generate_captcha(text='1'):
    """Generate a digit image."""
    capt = ImageCaptcha(width=32, height=32, font_sizes=[28])
    image_ = capt.generate_image(text)
    image_ = np.array(image_, dtype=np.uint8)
    return image_


if __name__ == '__main__':
    output_dir = 'captcha_images/images/train/'
    for i in range(50000):
        label = np.random.randint(0, 10)
        image = generate_captcha(str(label))
        image_name = 'image{}_{}.jpg'.format(i + 1, label)
        output_path = output_dir + image_name
        cv2.imwrite(output_path, image)
        if i % 100 == 0:
            print("create {} train images".format(i))
    output_dir = 'captcha_images/images/val/'
    for i in range(10000):
        label = np.random.randint(0, 10)
        image = generate_captcha(str(label))
        image_name = 'image{}_{}.jpg'.format(i + 1, label)
        output_path = output_dir + image_name
        cv2.imwrite(output_path, image)
        if i % 100 == 0:
            print("create {} val images".format(i))
