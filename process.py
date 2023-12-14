import numpy as np
import imageio.v3 as imageio
import cv2
import os


def RSGenerate(image, percent, colorization=True):
    #   RSGenerate(image,percent,colorization)
    #                               --Use to correct the color
    # image should be R G B format with three channels
    # percent is the ratio when restore whose range is [0,100]
    # colorization is True
    m, n, c = image.shape
    # print(np.max(image))
    image_normalize = image / np.max(image)
    image_generate = np.zeros(list(image_normalize.shape))
    if colorization:
        # Multi-channel Image R,G,B
        for i in range(c):
            image_slice = image_normalize[:, :, i]
            pixelset = np.sort(image_slice.reshape([m * n]))
            maximum = pixelset[np.floor(m * n * (1 - percent / 100)).astype(np.int32)]
            minimum = pixelset[np.ceil(m * n * percent / 100).astype(np.int32)]
            image_generate[:, :, i] = (image_slice - minimum) / (maximum - minimum + 1e-9)
            pass
        image_generate[np.where(image_generate < 0)] = 0
        image_generate[np.where(image_generate > 1)] = 1
        image_generate = cv2.normalize(image_generate, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image_generate.astype(np.uint8)


if __name__ == '__main__':
    if not os.path.exists('./result/rs/lr/'):
        os.makedirs('./result/rs/lr/')
    if not os.path.exists('./result/rs/hr/'):
        os.makedirs('./result/rs/hr/')

    for idx in range(240):
        image_lr = imageio.imread('./result/ori/lr/'+str(idx)+'.tif')
        image_hr = imageio.imread('./result/ori/hr/'+str(idx)+'.tif')
        image_lr_rs = RSGenerate(image_lr[:,:,[2,1,0]],1,1)
        image_hr_rs = RSGenerate(image_hr[:,:,[2,1,0]],1,1)
        imageio.imwrite('./result/rs/lr/' + str(idx) + '.tif',image_lr_rs)
        imageio.imwrite('./result/rs/hr/' + str(idx) + '.tif',image_hr_rs)
        

    