from math import floor
from .base import BaseDataset
from .base import BaseDataset
from .rotation import Rotation
from.tf_fn import train_tf_fn, test_tf_fn, tile_tf_fn, norm_tf_fn, to_t_tf_fn, to_i_tf_fn
import numpy as np
import torch
import torchvision.transforms as tf
import torchvision.utils as vutils
from PIL import Image


def norm_img(img):
    img_t = to_t_tf_fn(img)
    img_t = norm_tf_fn(img_t)
    return to_i_tf_fn(img_t)


class SemanticSensitiveRot(BaseDataset):
    def __init__(self, paths='', labels='', prob=float(0),
                 _max=-1, transformation_version=''):
        super().__init__(paths, labels, _max)
        self.prob = prob
        self.trans_v = transformation_version
        self.transformations = {
            'original_rotate': self.original_rotate,
            'lr_25_25': self.lr_25_25,
            'lr_25_5': self.lr_25_5,
            'lr_25_4': self.lr_25_4,
            'lr_25_3': self.lr_25_3,
            'remove_local_rotation_info':self.remove_local_rotation_info,
            'remove_patch_position_info': self.remove_patch_position_info,
            'deep_all':self.deep_all
        }

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img, n = self.transformations[self.trans_v](img, prob=self.prob)
        return to_t_tf_fn(img), n, label

    @staticmethod
    def remove_local_rotation_info(img, prob=float(0)):
        """
        1.rotate the image
        2.rotate the first patch of the rotated image by some degree, in order
        to let the total rotated degree of the first patch of the rotated image
        is 0

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1 - prob) / 3
        n = np.random.choice(np.arange(4), p=[prob, p_, p_, p_])
        # n = np.random.choice(np.arange(4), p=[0, 0, 1, 0])

        if n != 0:
            img = img.transpose(n + 1)
        wide = int(img.size[0] / 2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x + wide, y + wide)))

        ## this method can remove less information and give less noise
        ## since there is still some weak rule
        # n2 = np.random.choice(np.arange(4), p=[0.25, 0.25, 0.25, 0.25])
        # for i in range(3):
        #     imgs[(i+n2)%4] = imgs[(i+n2)%4].transpose(i+2)

        ## this method can remove more information and give move noise
        for i in range(4):
            n2 = np.random.choice(np.arange(4), p=[0.25, 0.25, 0.25, 0.25])
            if n2 == 0: continue
            imgs[i] = imgs[i].transpose(n2+1)



        # imgs[0] = imgs[0].transpose(5-n)
        # imgs[1] = imgs[1].transpose(5 - n)
        # imgs[2] = imgs[2].transpose(5 - n)
        # imgs[3] = imgs[3].transpose(5 - n)

        # for i in range(4):
        #     imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, n

    @staticmethod
    def remove_patch_position_info(img, prob=float(0)):
        """
        1.rotate 3 patches of the original image

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1 - prob) / 3
        n = np.random.choice(np.arange(4), p=[prob, p_, p_, p_])

        wide = int(img.size[0] / 2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x + wide, y + wide)))
        if n != 0:
            for i in range(1, 4):
                imgs[i] = imgs[i].transpose(n+1)

        for i in range(100):
            n2 = np.random.choice(np.arange(4), size=2, p=[0.25, 0.25, 0.25, 0.25])
            temp = imgs[n2[0]]
            imgs[n2[0]] = imgs[n2[1]]
            imgs[n2[1]] = temp




        # for i in range(4):
        #     imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))



        return img2, n

    @staticmethod
    def original_rotate(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/3
        n = np.random.choice(np.arange(4), p=[prob, p_, p_, p_])
        if n != 0:
            img = img.transpose(n+1)
        return img, n

    @staticmethod
    def lr_25_25(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

        if global_rot != 0:
            img = img.transpose(global_rot+1)

        wide = int(img.size[0]/2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x+wide, y+wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        # for i in range(4):
        #     imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, n

    @staticmethod
    def lr_25_3(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

        if global_rot != 0:
            img = img.transpose(global_rot+1)

        wide = int(img.size[0]/2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x+wide, y+wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        # for i in range(4):
        #     imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, patches+1

    @staticmethod
    def lr_25_4(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

        if global_rot != 0:
            img = img.transpose(global_rot+1)

        wide = int(img.size[0]/2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x+wide, y+wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        # for i in range(4):
        #     imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, local_rot+1

    @staticmethod
    def lr_25_5(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

        if global_rot != 0:
            img = img.transpose(global_rot+1)

        wide = int(img.size[0]/2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x+wide, y+wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        # for i in range(4):
        #     imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, global_rot+1

    @staticmethod
    def lr_nearest_3(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1 - prob) / 3
        n = np.random.choice([0, 10, 13, 20], p=[prob] + [p_ for i in range(3)])
        indice_dict = {0: 0, 10: 1, 13: 2, 20: 3}

        if n == 0:
            return img, 0

        global_rot = int((n - 1) / 6)
        patches = int((n - 1) / 3) % 2
        local_rot = (n - 1) % 3

        if global_rot != 0:
            img = img.transpose(global_rot + 1)

        wide = int(img.size[0] / 2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x + wide, y + wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        for i in range(4):
            imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))


        return img2, indice_dict[n]

    @staticmethod
    def deep_all(img, prob=float(0)):
        return img, 0



class SSRTrain(SemanticSensitiveRot):
    """Return tensor image with resize, normalize and other transform.

    """
    def __init__(self, paths='', labels='', prob=float(0), _max=-1, args=None):
        super().__init__(paths, labels, prob, _max, args.transformation_version)
        self.args = args

    def __getitem__(self, index):
        img_t, n, label = super().__getitem__(index)
        img = to_i_tf_fn(img_t)
        img = train_tf_fn(self.args)(img)
        img = tile_tf_fn(self.args)(img)
        img_t = to_t_tf_fn(img)
        return norm_tf_fn(img_t), n, label
        # return img_t, n, label


class SSRTest(SemanticSensitiveRot):
    """Return tensor image with resize and normalize.

    """
    def __init__(self, paths='', labels='', prob=float(0), _max=-1, args=None):
        super().__init__(paths, labels, prob, _max, args.transformation_version)
        self.args = args

    def __getitem__(self, index):
        img_t, n, label = super().__getitem__(index)
        img = to_i_tf_fn(img_t)
        img = test_tf_fn(self.args)(img)
        img_t = to_t_tf_fn(img)
        return norm_tf_fn(img_t), n, label
        # return img_t, n, label


class SSRTest2(SemanticSensitiveRot):
    """Return tensor image with resize and normalize.

    """
    def __init__(self, paths='', labels='', prob=float(0), _max=-1, args=None):
        super().__init__(paths, labels, prob, _max, args.transformation_version)
        self.args = args

    def __getitem__(self, index):
        img_t, n, label = super().__getitem__(index)
        img = to_i_tf_fn(img_t)
        img = test_tf_fn(self.args)(img)
        img_t = to_t_tf_fn(img)
        return norm_tf_fn(img_t), n, label, img_t