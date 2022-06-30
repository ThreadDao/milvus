import multiprocessing
import random
import time

import pandas as pd
from pymilvus import connections

from base.collection_wrapper import ApiCollectionWrapper
from common import common_func as cf
from utils.util_log import test_log as log

host = "10.98.0.4"


class TestParallelInsert:
    def test_parallel_insert(self):
        collection_w = ApiCollectionWrapper()
        # nb = 15000000
        # ni = 10000
        nb = 1000000
        ni = 10000
        process_num = 4
        nb_per_process = nb // process_num
        index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 128}}
        search_params = {"metric_type": "L2", "params": {"nprobe": 32}}

        connections.connect(host=host, port=19530)
        id_field = cf.gen_int64_field(name="feedid", is_primary=True, auto_id=False)
        time_field = cf.gen_int64_field(name="feed_time", is_primary=False)
        embedding_field = cf.gen_float_vec_field(name="embedding", dim=1024)
        schema = cf.gen_collection_schema(fields=[id_field, time_field, embedding_field])
        collection_w.init_collection(name="test_issue_17711", schema=schema, shards_num=9)
        collection_w.create_index('embedding', index_params)

        def do_insert(start):
            for _ in range(nb_per_process // ni):
                id_values = pd.Series(data=[i for i in range(start, start + ni)])
                time_values = pd.Series(data=[random.randint(0, 17000000) for _ in range(0, ni)])
                embedding_values = cf.gen_vectors(nb=ni, dim=1024)
                df = pd.DataFrame({
                    "feedid": id_values,
                    "feed_time": time_values,
                    'embedding': embedding_values
                })
                start += ni
                collection_w.insert(df, timeout=60)

        def do_search():
            connections.connect(host=host, port=19530)
            collection_w.init_collection(name="test_issue_17711")
            log.debug(collection_w.num_entities)
            collection_w.load()
            for _ in range(50):
                search_res, _ = collection_w.search(cf.gen_vectors(nb=1, dim=1024), 'embedding', search_params, 5)
                collection_w.query(expr=f'feedid in {search_res[0].ids}')

        insert_process = []
        for i in range(process_num):
            p = multiprocessing.Process(target=do_insert, args=(i * ni,))
            p.start()
            insert_process.append(p)

        search_process = []
        for i in range(2):
            p = multiprocessing.Process(target=do_search, args=())
            p.start()
            search_process.append(p)
        for p in insert_process:
            p.join()

        for p in search_process:
            p.join()

    def test_debug(self):
        search_params = {"metric_type": "L2", "params": {"nprobe": 32}}
        collection_w = ApiCollectionWrapper()

        def do_search():
            connections.connect(host=host, port=19530)
            collection_w.init_collection(name="test_issue_17711")
            log.debug(collection_w.num_entities)
            for _ in range(300):
                search_res, _ = collection_w.search(cf.gen_vectors(nb=1, dim=1024), 'embedding', search_params, 5)
                collection_w.query(expr=f'feedid in {search_res[0].ids}')

        process_list = []
        for i in range(2):
            p = multiprocessing.Process(target=do_search, args=())
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
