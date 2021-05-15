import torch


def save_model(model, file_path):
    print(f'Saving the model at:{file_path}')
    torch.save(model, file_path)


def load_model(file_path):
    print(f'Loading the model from:{file_path}')
    return torch.load(file_path)
