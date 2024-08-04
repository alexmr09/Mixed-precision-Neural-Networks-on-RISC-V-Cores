import init_utils
import common

# Initialize the environment and get the name
name = init_utils.initialize_environment(__file__)
args = init_utils.get_args()

# Set arguments from command line
max_acc_drop = args.max_acc_drop
device = args.device
method = args.method

import mpq_quantize
import torchvision.transforms.functional as TF
from PIL import Image
import pathlib
import numpy as np
import re
import define_network as define_network
from mcunet.model_zoo import build_model
from sklearn.model_selection import train_test_split

def tf_pre_process(img, normalize = True, form = 'Tensor'):
    img = TF.resize(img,(80, 80))
    img = TF.to_tensor(img) 
    if(normalize):
        img = TF.normalize(img, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    
    if(form == 'Numpy'):
        img = img.numpy()
    return img

images_dir_path_lab0 = pathlib.Path('mcunet_vww/dataset/vww-s256/val/0')
images_dir_path_lab1 = pathlib.Path('mcunet_vww/dataset/vww-s256/val/1')

images_path_0 = sorted(list(images_dir_path_lab0.glob('*.jpg')))
images_path_1 = sorted(list(images_dir_path_lab1.glob('*.jpg')))

images_path = images_path_0 + images_path_1

images_np_array = np.zeros((len(images_path), 3, 80, 80))
labels = np.zeros(len(images_path))

for i, image_path in enumerate(images_path):
    # Open the image file
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = tf_pre_process(image)
    label_path = re.split(r'[/\\]', str(image_path))
    labels[i] = label_path[-2]
    images_np_array[i,:,:,:] = image

X_train, X_test, y_train, y_test = train_test_split(images_np_array, labels, test_size = 2000, 
                                                    shuffle = True)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.50, 
                                                    shuffle = True)

BATCH_SIZE = 32
epochs = 5
lr = 0.0001

train_loader, val_loader, test_loader = mpq_quantize.create_dataloaders(BATCH_SIZE, X_train, y_train, X_test, y_test,
                                            X_val, y_val)

net_float, image_size, description = build_model(net_id = 'mcunet-vww1', pretrained = True)

mpq_quantize.fp_evaluate(net_float, test_loader, device)

## We will get rid of the Batch Normalization Layers and re-evaluate the Model's Accuracy

net = define_network.ProxylessNASNets_No_Bn()

net = define_network.merge_bn_network(net_float, net)

common.create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, 
                X_val = X_val, y_val = y_val, pretrained = True, 
                BATCH_SIZE = BATCH_SIZE, method = method, epochs = epochs, 
                lr = lr, max_acc_drop = max_acc_drop)
