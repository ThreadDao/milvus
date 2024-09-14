import os
import random

import numpy as np
from faker import Faker
from pymilvus import DataType

from base.client_base import TestcaseBase
from base.schema_wrapper import ApiFieldSchemaWrapper, ApiCollectionSchemaWrapper
from common import common_func as cf
from common import common_type as ct
from utils.util_log import test_log as log

pk_name = "pk"
vec_name = "vec"
dim = 128
fake = Faker()

def loop_gen_file_name(max_id=100):
    for i in range(3, max_id):
        file_name = "scalar_%05d.npy" % i
        yield f"/test/milvus/scalar/laion2b_json/{file_name}"


def check_file_exist(file_dir):
    if not os.path.isfile(file_dir):
        msg = "[check_file_exist] File not exist:{}".format(file_dir)
        log.error(msg)
        return False
    return True


def read_npy_file(file_name: str):
    log.info(file_name)
    if check_file_exist(file_name):
        try:
            file_data = np.load(file_name, allow_pickle=True)
            return file_data
        except Exception as e:
            log.error(f"[read_npy_file] Can not read parquet file: {e}")


def convert_data(chunk):
    'Converts a pandas dataframe to be a simple list of tuples, formatted how the `upsert()` method in the Pinecone '
    ' Python client expects.'
    data = []
    for row in chunk:
        row[vec_name] = [random.random() for _ in range(dim)]
        data.append(row)
    return data


class PreInsert:
    def __init__(self):
        self.data = []
        self._loop_file = loop_gen_file_name()

    def loop_data(self, ni=500):
        if len(self.data) < ni:
            while True:
                _dynamic = read_npy_file(next(self._loop_file))
                self.data.extend(convert_data(_dynamic))
                if len(self.data) >= ni:
                    break
        _v = self.data[:ni]
        self.data = self.data[ni:]
        return _v


class TestDynamic(TestcaseBase):
    def test_prepare_dynamic_data(self):
        # c_name = cf.gen_unique_str("debug_dynamic")
        c_name = "dynamic_collection"
        log.info(f"collection name {c_name}")

        # prepare schema
        pk_field, _ = ApiFieldSchemaWrapper().init_field_schema(name=pk_name, dtype=DataType.INT64, is_primary=True)
        vec_field, _ = ApiFieldSchemaWrapper().init_field_schema(name=vec_name, dtype=DataType.FLOAT_VECTOR, dim=128)
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=[pk_field, vec_field], auto_id=True, enable_dynamic_field=True)
        collection_w = self.init_collection_wrap(name=c_name, schema=schema)

        # insert data
        batch = 1000
        # batch = 200
        nb = 4000
        # nb = 5000000
        ni_cunt = int(nb / batch)

        num_entities = 0
        for i in range(0, ni_cunt):
            _data = []
            for _ in range(batch):
                row = {
                    vec_name: [random.random() for _ in range(dim)],
                    "x": fake.texts(nb_texts=200),
                    "y": fake.texts(nb_texts=200)
                }
                _data.append(row)
            insert_res, _ = collection_w.insert(_data)
            num_entities += insert_res.insert_count
            log.info(f"inserting num {num_entities}")

        # flush -> index -> load
        collection_w.flush()
        index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 96}}
        collection_w.create_index(vec_name, index_params, timeout=3600)
        collection_w.load()
        # collection_w.search