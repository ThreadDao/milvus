import time

import pandas as pd

from pymilvus import connections
from base.collection_wrapper import ApiCollectionWrapper
from base.schema_wrapper import ApiCollectionSchemaWrapper
from utils.util_log import test_log as log
from common import common_func as cf
from common import common_type as ct


class TestInsertFlush:
    """ Test case of Insert interface """

    def test_insert_flush(self):
        """
        target: test insert DataFrame data
        method: 1.create collection
                2.insert dataframe data
        expected: assert num entities
        """
        nb = 1000000
        ni = 1000
        dim = 128

        #  connect
        connections.connect(host=cf.param_info.param_host, port=cf.param_info.param_port)

        # create collection with 1 shard and auto id

        c_name = cf.gen_unique_str("insert_flush")
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field(dim=dim)]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, description=ct.default_desc,
                                                                        primary_field=ct.default_int64_field_name,
                                                                        auto_id=True)
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=c_name, schema=schema, shards_num=1)

        # create index
        index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}
        collection_w.create_index(ct.default_float_vec_field_name, index_params, index_name=ct.default_index_name)

        # load
        collection_w.load()

        # insert for loop
        ni_count = nb // ni
        for i in range(ni_count):
            df = pd.DataFrame({ct.default_float_vec_field_name: cf.gen_vectors(ni, dim)})

            # insert and flush
            _start_insert = time.time()
            collection_w.insert(data=df)
            collection_w.flush()

            log.info(f"Insert and flush {nb} cost {time.time() - _start_insert}s")
            log.info(f"collection num_entities: {collection_w.num_entities_without_flush}")

        log.info(f'Insert collection {collection_w.name} success with entities {collection_w.num_entities}')
