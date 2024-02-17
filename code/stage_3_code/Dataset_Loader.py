import torch.utils.data

from code.base_class.dataset import dataset
import pickle
import torchvision.transforms as transforms

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        train_data = []
        transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for sample in data['train']:
            image_path = sample['image']
            normal_image = transform_norm(image_path)
            sample['image'] = normal_image
        for sample in data['test']:
            image_path = sample['image']
            normal_image = transform_norm(image_path)
            sample['image'] = normal_image

        # print(data['train'][0])
        # print(data['train'][0]['image'].shape)
        train_data = torch.utils.data.DataLoader(data['train'], shuffle=True)
        test_data = torch.utils.data.DataLoader(data['test'], shuffle=False)
        return {'train_data': train_data, 'test_data': test_data}