import os 
import cv2
import random

"""Class for image-preprocessing - fetching images from files and distributing them into training-test-split sets"""
class Preprocessing():
    #checking the total number of folders and files(images) before the train-test-valid-split
    def dataset_metadata(self, dataset_path):
        """ This code navigates through dataset(s) and lists all the length of all folders and image-files present in them."""
        os.chdir(dataset_path)
        folders = os.listdir()
        folders_path = os.getcwd()
        print('[TOTAL FOLDERS]:', len(os.listdir()))
        total_file_count = 0 
        for index, folder in enumerate(folders): 
            if os.path.isdir(folders_path + f"/{folder}"):
                os.chdir(folders_path + f"/{folder}")
            else:
                continue
            print(f'[TOTAL IMAGES IN FOLDER {folder}]:', len(os.listdir()))
            total_file_count += len(os.listdir())
        print('[TOTAL IMAGE COUNT]:', total_file_count, f'WITH {len(folders)} CLASSES')
        root_dir = os.path.join(os.getcwd(), '..', '..', '..')
        os.chdir(root_dir)

    #checking image dimensions
    def image_metadata(self, image_path):
        """ image dimensions for mini-ImageNet is: (84, 84, 3) || image dimensions for EuroSAT_RGB is: (64, 64, 3) """
        image = cv2.imread(image_path)
        print(image.shape)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    #for extracting images as a dictionary of folder index as keys and list of images in that folder as its value
    # def get_images(self, dataset_path):
    #     print(dataset_path)
    #     images_path_dict = {}
    #     os.chdir(dataset_path)
    #     folders = os.listdir()
    #     folders_path = os.getcwd()
    #     for folder in folders: 
    #         if os.path.isdir(folders_path + f"/{folder}"):
    #             os.chdir(folders_path + f"/{folder}")
    #             images_path_list = [os.getcwd() + '/' + path for path in os.listdir()]
    #             images_path_dict[int(folder)] = images_path_list
    #         else:
    #             continue
    #     return images_path_dict

    def get_images(self, dataset_path):
        images_path_dict = {}
        os.chdir(dataset_path)
        folders = os.listdir()
        folders_path = os.getcwd()

        for folder in folders:
            if os.path.isdir(os.path.join(folders_path, folder)):
                os.chdir(os.path.join(folders_path, folder))
                images_path_list = [os.path.join(os.getcwd(), path) for path in os.listdir()]
                images_path_dict[int(folder)] = images_path_list
            else:
                continue
        return images_path_dict
    
    #taking a list of input images and randomly distributes them between train-test-valid datasets
    def train_test_valid_split(self, all_images):
    # random Sampling 
        data = []
        train_ratio = 0.75
        valid_ratio = 0.1

        for label, image_collection in all_images.items():
            for img in image_collection:
                data.append((label, img))
        random.seed(42)
        random.shuffle(data)

        train_split = int(train_ratio * len(data))
        valid_split = int(valid_ratio * len(data))

        train_data = data[:train_split]
        valid_data = data[train_split: train_split + valid_split]
        test_data = data[train_split + valid_split:]
        
        return train_data, valid_data, test_data

    def euroSat_get_categories(self, eurosat_path):
        os.chdir(eurosat_path)
        # random_categories = []
        data = []
        training_list = []
        testing_list = []
        random_categories = random.sample(os.listdir(), 5)

        for category in random_categories:
            image_counter  = 25
            select_five = []
            while image_counter:
                image = random.randint(0, len(os.listdir(os.getcwd() + "/" + category)) - 1)
                if os.listdir(os.getcwd() + "/" + category)[image] in data:
                    pass
                else: 
                    data.append((int(category), (f'{eurosat_path}/{category}/{os.listdir(os.getcwd() + "/" + category)[image]}')))
                    select_five.append((int(category), (f'{eurosat_path}/{category}/{os.listdir(os.getcwd() + "/" + category)[image]}')))
                    image_counter -= 1
            training_list.extend(random.sample(select_five, 5))
        testing_list = list(set(data).difference(set(training_list)))
        return training_list, testing_list[:75]

# obj = Preprocessing()
# train, test = obj.euroSat_get_categories('DL-FinalProject/EuroSAT_RGB')
# print(obj.train_test_valid_split(obj.get_images('DL-FinalProject/mini-ImageNet')))
                    
