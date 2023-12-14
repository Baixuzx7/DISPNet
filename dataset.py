import argparse
import torch.nn
import imageio
import os
import glob
import cv2
import torch.utils.data
import torchvision


parser = argparse.ArgumentParser()
# Train
parser.add_argument('--train_pan_size', type=int, default=256)
parser.add_argument('--train_ms_size', type=int, default=64)
parser.add_argument('--train_pan_stride', type=int, default=128)
parser.add_argument('--train_ms_stride', type=int, default=32)
# Test
parser.add_argument('--test_pan_size',type=int,default=256)
parser.add_argument('--test_ms_size',type=int,default=64)
parser.add_argument('--test_pan_stride',type=int,default=256)
parser.add_argument('--test_ms_stride',type=int,default=64)

opt = parser.parse_args()

# Preparation : A directory Named : Dataset (Contains Two Directories named pan_label, ms_label)
# There is only a picture in the directory named 1.tif
# First, Down Sample the images in the pan_label and ms_label to get the pan and ms
# Down-Sample Method : Bicubic
# CreateDataset(datapath = './Dataset/pan_label',savedir = './Dataset/pan')
# CreateDataset(datapath = './Dataset/ms_label',savedir = './Dataset/ms')
def CreateDataset(datapath, savedir):
    image_label = imageio.imread(datapath + '/1.tif')
    m, n = image_label.shape[0], image_label.shape[1]
    image = cv2.resize(image_label, dsize=(n // 4, m // 4), interpolation=cv2.INTER_CUBIC)
    print(image_label.shape)
    print(image.shape)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    imageio.imwrite(savedir + '/1.tif', image)


# Preparation: A directory Named : Dataset (Contains 4 directories)
# ./Dataset
#   .../pan_label
#       .../1.tif
#   .../pan
#       .../1.tif
#   .../ms_label
#       .../1.tif
#   .../ms
#       .../1.tif
# GenerateIMageFolder(data_dir = './Dataset/pan_label',
#                     save_dir = './IMageFolder/pan_label',
#                     image_size = opt.pan_size * 4,
#                     image_stride = opt.pan_stride * 4,
#                     Filestyle = '.tif',
#                     mode = 'GRAYSCALE')
# GenerateIMageFolder(data_dir = './Dataset/pan',
#                     save_dir = './IMageFolder/pan',
#                     image_size = opt.pan_size,
#                     image_stride = opt.pan_stride,
#                     Filestyle = '.tif',
#                     mode = 'GRAYSCALE')
# GenerateIMageFolder(data_dir = './Dataset/ms_label',
#                     save_dir = './IMageFolder/ms_label',
#                     image_size = opt.ms_size * 4,
#                     image_stride = opt.ms_stride * 4,
#                     Filestyle = '.tif',
#                     mode = 'COLOR')
# GenerateIMageFolder(data_dir = './Dataset/ms',
#                     save_dir = './IMageFolder/ms',
#                     image_size = opt.ms_size,
#                     image_stride = opt.ms_stride,
#                     Filestyle = '.tif',
#                     mode = 'COLOR')
def GenerateIMageFolder(data_dir, save_dir, image_size, image_stride, Filestyle, mode,flag = True):
    # Filestyle,the image format like .jpeg,.png.tif and so on
    def prepare_data(dataset, Filestyle):
        data_dir = os.path.join(os.getcwd(), dataset)
        Fileformat = '*' + Filestyle
        print(Fileformat)
        data = glob.glob(os.path.join(data_dir, Fileformat))
        # 将图片按序号排序
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
        return data

    def make_data(sub_input_sequence, data_dir, Filestyle):
        savepath = data_dir
        length = len(sub_input_sequence)
        for i in range(length):
            print('Rate of Process : [{}/{}]'.format(i, length))
            image = sub_input_sequence[i]
            if len(list(image.shape)) == 2:
                m, n = image.shape
                image = image.reshape([m, n])
                imageio.imwrite(savepath + '/' + str(i) + Filestyle, image)
            else:
                m, n, c = image.shape
                image = image.reshape([m, n, c])
                imageio.imwrite(savepath + '/' + str(i) + Filestyle, image)
            pass

    def input_setup(data_dir, save_dir, image_size, image_stride, Filestyle, train = True):
        data = prepare_data(data_dir, Filestyle)
        sub_input_sequence = []
        length = len(data)
        if mode == 'COLOR':
            for j in range(length):
                print('Segment Image [{}/{}]'.format(j, length))
                image_input = imageio.imread(data[j])
                h, w, c = image_input.shape
                if train:
                    print('Training')
                    input_ = image_input[:h//4 * 3,:,:]
                else:
                    print('Testing')
                    input_ = image_input[h//4 * 3:,:,:]
                h, w, c = input_.shape
                for x in range(0, h - image_size + 1, image_stride):
                    for y in range(0, w - image_size + 1, image_stride):
                        sub_input = input_[x:x + image_size, y:y + image_size, :]
                        sub_input = sub_input.reshape([image_size, image_size, c])
                        sub_input_sequence.append(sub_input)
                        pass
                    pass
        elif mode == 'GRAYSCALE':
            for j in range(length):
                print('Segment Image [{}/{}]'.format(j, length))
                image_input = imageio.imread(data[j])
                h, w = image_input.shape
                if train:
                    print('Training')
                    input_ = image_input[:h // 4 * 3, :]
                else:
                    print('Testing')
                    input_ = image_input[h // 4 * 3:, :]
                h, w = input_.shape
                for x in range(0, h - image_size + 1, image_stride):
                    for y in range(0, w - image_size + 1, image_stride):
                        sub_input = input_[x:x + image_size, y:y + image_size]
                        sub_input = sub_input.reshape([image_size, image_size])
                        sub_input_sequence.append(sub_input)
                        pass
                    pass
        make_data(sub_input_sequence, save_dir, Filestyle)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        pass
    print(save_dir, 'Creating----->Processing')
    input_setup(data_dir, save_dir, image_size, image_stride, Filestyle, train=flag)


class PanSharpeningDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.images_pan = list(sorted(os.listdir(os.path.join(self.root, 'pan'))))
        self.images_pan.sort(key=lambda x: int(x[:-4]))
        self.images_ms = list(sorted(os.listdir(os.path.join(self.root, 'ms'))))
        self.images_ms.sort(key=lambda x: int(x[:-4]))
        self.images_ms_label = list(sorted(os.listdir(os.path.join(self.root, 'ms_label'))))
        self.images_ms_label.sort(key=lambda x: int(x[:-4]))
        self.images_pan_label = list(sorted(os.listdir(os.path.join(self.root, 'pan_label'))))
        self.images_pan_label.sort(key=lambda x: int(x[:-4]))

    def __getitem__(self, item):
        image_pan_path = os.path.join(self.root, 'pan', self.images_pan[item])
        image_ms_path = os.path.join(self.root, 'ms', self.images_ms[item])
        image_ms_label_path = os.path.join(self.root, 'ms_label', self.images_ms_label[item])
        image_pan_label_path = os.path.join(self.root, 'pan_label', self.images_pan_label[item])

        image_pan = imageio.imread(image_pan_path)
        image_ms = imageio.imread(image_ms_path)  
        image_ms_label = imageio.imread(image_ms_label_path)  
        image_pan_label = imageio.imread(image_pan_label_path)

        if self.transform is not None:
            image_pan = self.transform(image_pan)
            image_ms = self.transform(image_ms)
            image_pan_label = self.transform(image_pan_label)
            image_ms_label = self.transform(image_ms_label)
            pass

        return image_pan, image_ms, image_pan_label, image_ms_label

    def __len__(self):
        return len(self.images_ms_label)


if __name__ == '__main__':
    CreateDataset(datapath = './Dataset/pan_label',savedir = './Dataset/pan')
    CreateDataset(datapath = './Dataset/ms_label',savedir = './Dataset/ms')
    # Generate Train Dataset
    GenerateIMageFolder(data_dir = './Dataset/pan_label',
                        save_dir = './data/TrainFolder/pan_label',
                        image_size = opt.train_pan_size * 4,
                        image_stride = opt.train_pan_stride * 4,
                        Filestyle = '.tif',
                        mode = 'GRAYSCALE',
                        flag = True)
    GenerateIMageFolder(data_dir = './Dataset/pan',
                        save_dir = './data/TrainFolder/pan',
                        image_size = opt.train_pan_size,
                        image_stride = opt.train_pan_stride,
                        Filestyle = '.tif',
                        mode = 'GRAYSCALE',
                        flag = True)
    GenerateIMageFolder(data_dir = './Dataset/ms_label',
                        save_dir = './data/TrainFolder/ms_label',
                        image_size = opt.train_ms_size * 4,
                        image_stride = opt.train_ms_stride * 4,
                        Filestyle = '.tif',
                        mode = 'COLOR',
                        flag = True)
    GenerateIMageFolder(data_dir = './Dataset/ms',
                        save_dir = './data/TrainFolder/ms',
                        image_size = opt.train_ms_size,
                        image_stride = opt.train_ms_stride,
                        Filestyle = '.tif',
                        mode = 'COLOR',
                        flag = True)
    # Generate Test Dataset
    GenerateIMageFolder(data_dir = './Dataset/pan_label',
                        save_dir = './data/TestFolder/pan_label',
                        image_size = opt.test_pan_size * 4,
                        image_stride = opt.test_pan_stride *  4,
                        Filestyle = '.tif',
                        mode = 'GRAYSCALE',
                        flag = False)
    GenerateIMageFolder(data_dir = './Dataset/pan',
                        save_dir = './data/TestFolder/pan',
                        image_size = opt.test_pan_size,
                        image_stride = opt.test_pan_stride,
                        Filestyle = '.tif',
                        mode = 'GRAYSCALE',
                        flag = False)
    GenerateIMageFolder(data_dir = './Dataset/ms_label',
                        save_dir = './data/TestFolder/ms_label',
                        image_size =  opt.test_ms_size * 4,
                        image_stride = opt.test_ms_stride * 4,
                        Filestyle = '.tif',
                        mode = 'COLOR',
                        flag = False)
    GenerateIMageFolder(data_dir = './Dataset/ms',
                        save_dir = './data/TestFolder/ms',
                        image_size = opt.test_ms_size,
                        image_stride = opt.test_ms_stride,
                        Filestyle = '.tif',
                        mode = 'COLOR',
                        flag = False)