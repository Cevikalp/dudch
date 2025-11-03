import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN

class Random300K_Images(torch.utils.data.Dataset):

    def __init__(self, file_path='/home/mlcv/CevikalpPy/NirvanaOSet/data/300K_random_images.npy', transform=None, extendable = False):
        self.data = np.load(file_path)
        if extendable:
            self.data = list(self.data)
        self.transform = transform
        self.mean = [0.519,0.511,0.508]
        self.std = [0.317,0.312,0.306]
        self.offset = 0
        
    def __getitem__(self, index):
        index = (index + self.offset) % self.__len__()
        img = self.data[index]
        img = Image.fromarray(img)
        if img.mode == 'L':
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 10  # 0 is the class

    def __len__(self):
        return len(self.data)

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, file_path, transform, exclude_cifar):

        data_file = open(file_path, "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open(exclude_cifar, 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs


    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)
	
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class
	
    def __len__(self):
        return 79302017

class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy() if hasattr(img,'numpy') else img, mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST_Filter(MNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)
    
    def __Noisy__(self, ratio):
        if (ratio>1 and ratio<=0):
            raise("ratio should be in range between 0. and 1.")
        targets = np.array(self.targets)
        num_classes = len(np.unique(targets))
        num_instance = len(self.data)
        n_instance_per_class = num_instance / num_classes
        n_instance_out_per_class = int(n_instance_per_class*ratio)
        # n_instance_in_per_class = n_instance_per_class - n_instance_out_per_class
        data_in, target_in = [], []
        data_out, target_out = [], []
        
        for i in range(num_classes):
            data_out.append(self.data[targets==i][:n_instance_out_per_class])
            target_out.append(targets[targets==i][:n_instance_out_per_class])
            data_in.append(self.data[targets==i][n_instance_out_per_class:])
            target_in.append(targets[targets==i][n_instance_out_per_class:])
        self.data_out = np.concatenate(data_out,axis=0)
        self.target_out = np.concatenate(target_out,axis=0)
        data_in = np.concatenate(data_in,axis=0)
        target_in = np.concatenate(target_in,axis=0)
        self.data, self.targets = data_in, target_in

class MNIST_OSR(object):
    def __init__(self, known, dataroot='/home/mlcv/CevikalpPy/NirvanaOSet/data/mnist', use_gpu=True, num_workers=0, batch_size=128, img_size=32, noisy_ratio=0.0):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pin_memory = True if use_gpu else False
        if(noisy_ratio == 0.0):
            trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
            print('All Train Data:', len(trainset))
            trainset.__Filter__(known=self.known)
        else:
            print("Noisy data experiments in business w/ %.2f"%noisy_ratio)
            trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
            print('All Train Data:', len(trainset))
            trainset.__Filter__(known=self.known)
            trainset.__Noisy__(noisy_ratio)
            self.noisy_data = trainset.data_out
            self.noisy_targets = trainset.target_out

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

    def __Noisy__(self, ratio):
        if (ratio>1 and ratio<=0):
            raise("ratio should be in range between 0. and 1.")
        targets = np.array(self.targets)
        num_classes = len(np.unique(targets))
        num_instance = len(self.data)
        n_instance_per_class = num_instance / num_classes
        n_instance_out_per_class = int(n_instance_per_class*ratio)
        # n_instance_in_per_class = n_instance_per_class - n_instance_out_per_class
        data_in, target_in = [], []
        data_out, target_out = [], []
        
        for i in range(num_classes):
            data_out.append(self.data[targets==i][:n_instance_out_per_class])
            target_out.append(targets[targets==i][:n_instance_out_per_class])
            data_in.append(self.data[targets==i][n_instance_out_per_class:])
            target_in.append(targets[targets==i][n_instance_out_per_class:])
        self.data_out = np.concatenate(data_out,axis=0)
        self.target_out = np.concatenate(target_out,axis=0)
        data_in = np.concatenate(data_in,axis=0)
        target_in = np.concatenate(target_in,axis=0)
        self.data, self.targets = data_in, target_in

class CIFAR10_OSR(object):
    def __init__(self, known, dataroot='/home/mlcv/CevikalpPy/NirvanaOSet/data/cifar10', use_gpu=True, num_workers=0, batch_size=128, img_size=32, noisy_ratio=0.0):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pin_memory = True if use_gpu else False
        if(noisy_ratio == 0.0):
            trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
            print('All Train Data:', len(trainset))
            trainset.__Filter__(known=self.known)
        else:
            print("Noisy data experiments in business w/ %.2f"%noisy_ratio)
            trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
            print('All Train Data:', len(trainset))
            trainset.__Filter__(known=self.known)
            trainset.__Noisy__(noisy_ratio)
            self.noisy_data = trainset.data_out
            self.noisy_targets = trainset.target_out

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR10_CSR(object):
    def __init__(self, dataroot='/dhome/mlcv/CevikalpPy/NirvanaOSet/ata/cifar10', use_gpu=True, num_workers=0, batch_size=128, img_size=32, known_class=None):
        self.num_classes = 10
        if known_class:
            self.num_classes = len(known_class)
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=known_class)
        print('All Train Data:', len(trainset))
    

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.trainset = trainset
        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=known_class)
        print('All Test Data:', len(testset))
        self.testset = testset
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset))
        print('All Test: ', (len(testset) ))

class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR(object):
    def __init__(self, known, dataroot='/dhome/mlcv/CevikalpPy/NirvanaOSet/ata/cifar100', use_gpu=True, num_workers=0, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pin_memory = True if use_gpu else False
        
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )


class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)
    
    def __Noisy__(self, ratio):
        if (ratio>1 and ratio<=0):
            raise("ratio should be in range between 0. and 1.")
        targets = np.array(self.labels)
        num_classes = len(np.unique(targets))
        num_instance = len(self.data)
        n_instance_per_class = num_instance / num_classes
        n_instance_out_per_class = int(n_instance_per_class*ratio)
        # n_instance_in_per_class = n_instance_per_class - n_instance_out_per_class
        data_in, target_in = [], []
        data_out, target_out = [], []
        
        for i in range(num_classes):
            data_out.append(self.data[targets==i][:n_instance_out_per_class])
            target_out.append(targets[targets==i][:n_instance_out_per_class])
            data_in.append(self.data[targets==i][n_instance_out_per_class:])
            target_in.append(targets[targets==i][n_instance_out_per_class:])
        self.data_out = np.concatenate(data_out,axis=0)
        self.target_out = np.concatenate(target_out,axis=0)
        data_in = np.concatenate(data_in,axis=0)
        target_in = np.concatenate(target_in,axis=0)
        self.data, self.labels = data_in, target_in

    

class SVHN_OSR(object):
    def __init__(self, known, dataroot='/home/mlcv/CevikalpPy/NirvanaOSet/data/svhn', use_gpu=True, num_workers=0, batch_size=128, img_size=32, noisy_ratio=0.0):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pin_memory = True if use_gpu else False
        if(noisy_ratio == 0.0):
            trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
            print('All Train Data:', len(trainset))
            trainset.__Filter__(known=self.known)
        else:
            print("Noisy data experiments in business w/ %.2f"%noisy_ratio)
            trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
            print('All Train Data:', len(trainset))
            trainset.__Filter__(known=self.known)
            trainset.__Noisy__(noisy_ratio)
            self.noisy_data = np.transpose(trainset.data_out, (0, 2, 3, 1))
            self.noisy_targets = trainset.target_out


        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot='/home/mlcv/CevikalpPy/NirvanaOSet/data/tiny_imagenet', use_gpu=True, num_workers=0, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))
# out_dataset = Random300K_Images()
# dataset = CIFAR10_Filter_Noisy('./data/cifar10', train=True)
# dataset.__Noisy__(0.2)
# print("stop here :)")

# dataset = MNIST_Filter(root='./data/mnist',train=True)
# dataset.__Noisy__(0.2)
# dataset.__getitem__(1)