import pickle


def load_pickle(file_path: str):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
