import custom_dataset
# import logger
dataset_path = './kaggle/blood_cell/'
train_data, test_data, valid_data = custom_dataset.parse_dataset(dataset_path)
_, test_queue = custom_dataset.preprocess_data(train_data, test_data, batch_size=8)