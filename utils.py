from tqdm import tqdm
from mat2vec.processing import MaterialsTextProcessor
import json
import pickle
import os


def read_json(filename: str):
    assert isinstance(filename, str)
    with open(filename) as json_data:
        data = json.loads(json_data.read())
    return data


def read_pickle(filename: str):
    assert isinstance(filename, str)
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


def write_pickle(x, filename: str):
    assert isinstance(filename, str)
    with open(filename, 'wb') as pickle_file:
        pickle.dump(x, pickle_file)


def write_file(x, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(x)


def read_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def preprocess(dataset_json: str, output_name='corpus'):
    dataset = read_json(dataset_json)
    text_corpus = [doc['description'] for doc in dataset]
    text_processor = MaterialsTextProcessor()
    clean_corpus = [text_processor.process(abstract) for abstract in tqdm(text_corpus)]

    corpus = '\n'.join([' '.join(item[0]) for item in clean_corpus])
    output = os.path.join('mat2vec', 'training', 'data', output_name)
    write_file(corpus, output)
