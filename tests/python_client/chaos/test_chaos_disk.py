import os
import time
import multiprocessing

import pytest
import h5py
import numpy as np
from pymilvus import connections, DataType

from base.collection_wrapper import ApiCollectionWrapper
from base.schema_wrapper import ApiCollectionSchemaWrapper, ApiFieldSchemaWrapper
from base.utility_wrapper import ApiUtilityWrapper
from common import common_func as cf
from common import common_type as ct

from utils.util_log import test_log as log

host = "10.100.32.154"
# host = "10.98.0.4"
port = 31326
# port = 19530
hdf5_source_file = "/Users/nausicca/Downloads/vectors/sift-128-euclidean.hdf5"
ni = 50000
nb = 1000000
"""
curl -fsSL -o chaosd-v1.0.0-linux-amd64.tar.gz https://mirrors.chaos-mesh.org/chaosd-$CHAOSD_VERSION-linux-amd64.tar.gz
"""


def chaos_disk():
    os.system('scripts/chaos_disk.sh')


class TestChaosDisk:

    @pytest.fixture(scope="function", autouse=True)
    def connection(self):
        connections.connect("default", host=host, port=port)

    def test_disk_during_insert(self):
        """
        target:
        method: chaosd attack disk fill -c 50 -d
        expected:
        """
        # define field and schema
        collection_w = ApiCollectionWrapper()
        filed_w = ApiFieldSchemaWrapper()
        schema_w = ApiCollectionSchemaWrapper()
        fields = [filed_w.init_field_schema(name="id", dtype=DataType.INT64, is_primary=True)[0],
                  filed_w.init_field_schema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128)[0]]
        schema = schema_w.init_collection_schema(fields, auto_id=True)[0]

        # create collection
        c_name = "issue_OuwtMbBt"
        collection_w.init_collection(name=cf.gen_unique_str("disk"), schema=schema, shards_num=2, timeout=20)
        # log.info(collection_w.num_entities)

        dataset = h5py.File(hdf5_source_file)

        # insert
        vectors = np.array(dataset['train'])
        # .astype(np.float(32))
        for i in range(5):
            s = time.time()
            for i in range(nb // ni):
                start = i * ni
                end = (i + 1) * ni
                # int_values = np.arange(start, end, dtype='int64')
                print(f'start insert {start}:{end}')
                data = [vectors[start: end]]
                collection_w.insert(data)
                log.info(collection_w.num_entities)
            log.info(f'{i} insert cost: {time.time() - s}')

    def test_disk_fill_during_index(self):
        c_name = "disk_JLJduoAk"
        vec_field_name = "vec"
        default_index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 512}}
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(c_name)
        collection_w.create_index(vec_field_name, default_index_params, timeout=1800)
        collection_w.indexes
        """
        disk_r2NHiyMw {'row_count': 56650000}
        disk_pem1gMUE {'row_count': 6550000}
        disk_wKYl6AJ8 {'row_count': 25350000}
        disk_JLJduoAk {'row_count': 100050000}
        disk_mt2YTTGx {'row_count': 10000000}
        disk_MgdoQXyb {'row_count': 87000000}
        disk_WjNQUIyW {'row_count': 9000000}
        disk_hZbh5k6Q {'row_count': 10000000}
        disk_3Y7MChs5 {'row_count': 16650000}
        """

    def test_disk_attack_etcd_ddl(self):
        collection_num = 50
        nb = 2000
        default_index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 64}}
        df = cf.gen_default_dataframe_data(nb=nb)

        for i in range(collection_num):
            log.info(f"Start ddl {i} collection")
            # Create collection
            utilityWrapper = ApiUtilityWrapper()
            collection_w = ApiCollectionWrapper()
            schema = cf.gen_default_collection_schema()
            collection_w.init_collection(cf.gen_unique_str("etcd"), schema=schema, shards_num=1)

            # insert
            collection_w.insert(df)
            assert collection_w.num_entities == nb

            # create index
            collection_w.create_index(ct.default_float_vec_field_name, default_index_params, timeout=1800)

            res, _ = utilityWrapper.wait_for_index_building_complete(collection_w.name)
            if res is True:
                log.debug(collection_w.indexes[0].params)

            # drop collection
            collection_w.drop()

    def test_delete_issue(self):
        # create collection
        log.info("Create collection")
        from pymilvus import Collection, CollectionSchema, FieldSchema
        fields = [FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                  FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=128)]
        schema = CollectionSchema(fields)
        c = Collection(name=cf.gen_unique_str("issue"), schema=schema, timeout=20)

        # insert nb entities
        nb = 33000
        dataset = h5py.File(hdf5_source_file)
        vectors = np.array(dataset['train'])
        ids = [i for i in range(nb)]
        data = [ids, vectors[0: nb].tolist()]
        log.info(f"Insert {nb} entities")
        insert_res = c.insert(data)
        log.info(f"Collection num entities: {c.num_entities}")
        log.info(f"Len of primary keys: {len(insert_res.primary_keys)}")

        log.info("Load collection")
        c.load()
        expr = f"id in {insert_res.primary_keys[:5]}"

        log.info(f"Query with expr {expr}")
        query_res = c.query(expr, consistency_level="Strong")
        log.info(f"Query result: {query_res}")
        if len(query_res) == 5:
            log.info(f"Delete with expr {expr}")
            del_res = c.delete(expr)
            log.info(f"Delete result: {del_res}")

        res = c.query(expr, consistency_level="Strong")
        log.info(f"Query result with {expr} after delete: {res}")

        expr_1 = f"id in [-1]"
        dd_res = c.delete(expr_1)
        log.info(dd_res)
