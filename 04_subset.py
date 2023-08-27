import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dict = unpickle('dat/data_batch_2')

# Extract first 10 images and labels
subset_data = data_dict[b'data'][:10]
subset_labels = data_dict[b'labels'][:10]

# Create a new dictionary for the subset
subset_dict = {
    b'batch_label': b'subset of data_batch_2',
    b'labels': subset_labels,
    b'data': subset_data,
    b'filenames': data_dict[b'filenames'][:10]  # extracting filenames for the 10 images
}

# Save to a new pickle file
with open('dat/data_subset_10.pkl', 'wb') as f:
    pickle.dump(subset_dict, f)

print("Subset created and saved!")
