import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dict = unpickle('dat/data_subset_10.pkl')
images_data = data_dict[b'data']

# The number of images in data_batch_2
num_images = images_data.shape[0]
print(f"Number of images in data_batch_2: {num_images}")
