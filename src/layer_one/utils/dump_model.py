import os
import pickle


def dump_pickle(file_obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'bw') as file:
        pickle.dump(file_obj, file)
