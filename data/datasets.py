import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
import cv2
from torch.utils.data import DistributedSampler

random.seed(42)


# several data augumentation strategies
def cv_random_flip(img, depth, label, edge):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, depth, label, edge


def randomCrop(image, depth, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), depth.crop(random_region), label.crop(random_region), edge.crop(random_region)


def randomRotation(image, depth, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, depth, label, edge


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


# def randomPeper(img):
#     img = np.array(img)
#     noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
#     for i in range(noiseNum):
#
#         randX = random.randint(0, img.shape[0] - 1)
#
#         randY = random.randint(0, img.shape[1] - 1)
#
#         if random.randint(0, 1) == 0:
#
#             img[randX, randY] = 0
#
#         else:
#
#             img[randX, randY] = 255
#     return Image.fromarray(img)
def randomPeper_eg(img, edge):
    img = np.array(img)
    edge = np.array(edge)

    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0
            edge[randX, randY] = 0

        else:

            img[randX, randY] = 255
            edge[randX, randY] = 255

    return Image.fromarray(img), Image.fromarray(edge)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, depth_root, gt_root, edge_root, imgsize):
        self.trainsize = imgsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.bmp')
                      or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)

        self.filter_files()
        self.size = len(self.images)
        self.kernel = np.ones((5, 5), np.uint8)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)),
             transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.rgb_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])

        # edge = self.binary_loader(self.edges[index])

        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge)

        image, depth, gt, edge = cv_random_flip(image, depth, gt, edge)
        image, depth, gt, edge = randomCrop(image, depth, gt, edge)
        image, depth, gt, edge = randomRotation(image, depth, gt, edge)
        # image = colorEnhance(image)
        # depth = colorEnhance(depth)

        gt, edge = randomPeper_eg(gt, edge)

        image = self.img_transform(image)
        depth = self.depth_transform(depth)
        gt = self.gt_transform(gt)
        edge = self.edge_transform(edge)

        return image, depth, gt, edge

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.edges) and len(self.edges) == len(
            self.depths)
        images = []
        depths = []
        gts = []
        edges = []
        for img_path, depth_path, gt_path, edge_path in zip(self.images, self.depths, self.gts, self.edges):
            img = Image.open(img_path)
            depth = Image.open(depth_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size and gt.size == edge.size and depth.size == img.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.depths = depths
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, depth, gt, edge):
        assert img.size == gt.size and gt.size == edge.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return (img.resize((w, h), Image.BILINEAR),
                    depth.resize((w, h), Image.BILINEAR),
                    gt.resize((w, h), Image.NEAREST),
                    edge.resize((w, h), Image.NEAREST))
        else:
            return img, depth, gt, edge

    def __len__(self):
        return self.size


class test_dataset:
    """load test dataset (batchsize=1)"""

    def __init__(self, image_root, depth_root, gt_root, testsize=384):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image0 = self.rgb_loader(self.images[self.index])
        depth0 = self.rgb_loader(self.depths[self.index])
        image = self.transform(image0).unsqueeze(0)
        depth = self.transform(depth0).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        return image, depth, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_dataset_eval:
    """load test dataset (batchsize=1)"""

    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        # image = self.rgb_loader(self.images[self.index])
        image = self.binary_loader(os.path.join(self.image_root, self.img_list[self.index] + '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root, self.img_list[self.index] + '.png'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
