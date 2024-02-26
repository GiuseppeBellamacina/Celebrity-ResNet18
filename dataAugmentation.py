import os
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import functional as TF
from tqdm import tqdm
from random import randint, uniform

def hflip(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Horizontal Flipping'):
        new_dataset.append((TF.hflip(image), label))
    return new_dataset

def vflip(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Vertical Flipping'):
        new_dataset.append((TF.vflip(image), label))
    return new_dataset

def rotate(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Rotating'):
        new_dataset.append((TF.rotate(image, randint(-30,30)), label))
    return new_dataset

def leftrotate(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Rotating'):
        new_dataset.append((TF.rotate(image, -90), label))
    return new_dataset

def rightrotate(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Rotating'):
        new_dataset.append((TF.rotate(image, 90), label))
    return new_dataset

def colorbright(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Color Jittering'):
        new_dataset.append((TF.adjust_brightness(image, uniform(0.5,1.5)), label))
    return new_dataset

def colorcontrast(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Color Jittering'):
        new_dataset.append((TF.adjust_contrast(image, uniform(0.5,1.5)), label))
    return new_dataset

def colorsaturation(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Color Jittering'):
        new_dataset.append((TF.adjust_saturation(image, uniform(0.5,1.5)), label))
    return new_dataset

def colorhue(dataset):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Color Jittering'):
        new_dataset.append((TF.adjust_hue(image, uniform(-0.5,0.5)), label))
    return new_dataset

def grayscale(dataset): # prende in input un'immagine e torna un'immagine
    new_dataset = []
    for image, label in tqdm(dataset, desc='Grayscaling'):
        image = TF.to_pil_image(image)
        new_dataset.append((TF.to_grayscale(image), label))
    return new_dataset

def perspective(dataset, start, end):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Perspective'):
        new_dataset.append((TF.perspective(image, start, end), label))
    return new_dataset

def resize(dataset, size):
    new_dataset = []
    for image, label in tqdm(dataset, desc='Resizing'):
        new_dataset.append((TF.resize(image, size, antialias=True), label))
    return new_dataset

def saveDataset(dataset, save_path, classes):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for image, label in tqdm(dataset, desc=f'Saving on {save_path}'):
        new_save_path = save_path + '/' + classes[label]
        if not os.path.exists(new_save_path):
            os.mkdir(new_save_path)
        if type(image) == type(Tensor()):
            image = TF.to_pil_image(image)
        image.save(os.path.join(new_save_path, str(randint(0, 10000000)) + '.jpg'))

# Main
def main():
    data_dir = './celebrityData'
    dataset = ImageFolder(data_dir, transform=ToTensor())
    classes = dataset.classes
    out_dir = './celebrityDataAugmented'
    resize_dir = './celebrityDataResized'

    dataset = resize(dataset, (474, 474))
    saveDataset(dataset, out_dir, classes)
    saveDataset(dataset, resize_dir, classes)
    new = hflip(dataset)
    saveDataset(new, out_dir, classes)
    new = vflip(dataset)
    saveDataset(new, out_dir, classes)
    new = rotate(dataset) # 20-40 degrees
    saveDataset(new, out_dir, classes)
    new = leftrotate(dataset)
    saveDataset(new, out_dir, classes)
    new = rightrotate(dataset)
    saveDataset(new, out_dir, classes)
    new = colorbright(dataset) # 0.5-1.5 brightness
    saveDataset(new, out_dir, classes)
    new = colorcontrast(dataset) # 0.5-1.5 contrast
    saveDataset(new, out_dir, classes)
    new = colorsaturation(dataset) # 0.5-1.5 saturation
    saveDataset(new, out_dir, classes)
    new = colorhue(dataset) # -0.5-0.5 hue
    saveDataset(new, out_dir, classes)
    new = grayscale(dataset)
    saveDataset(new, out_dir, classes)

    new = perspective(dataset,
        Tensor([[0, 0], [0, 224], [224, 224], [224, 0]]),
        Tensor([[0, 0], [0, 224], [224, 192], [224, 32]])
    )
    saveDataset(new, out_dir, classes)
    new = perspective(dataset,
        Tensor([[0, 0], [0, 224], [224, 224], [224, 0]]),
        Tensor([[0, 32], [0, 192], [224, 224], [224, 0]])
    )
    saveDataset(new, out_dir, classes)
    print("\33[1;32mAugmentation Done!\33[0m")

if __name__ == '__main__':
    main()