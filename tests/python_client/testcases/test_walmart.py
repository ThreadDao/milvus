import random
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import pytest
from pymilvus import connections

from base.collection_wrapper import ApiCollectionWrapper
from common import common_func as cf
from common import common_type as ct

from base.client_base import TestcaseBase
from utils.util_log import test_log as log

coll_name = "test_walmart"
partition_num = 450
dim = 384


class TestIssue(TestcaseBase):
    """ test reproduce walmart issue"""

    def teardown_method(self, method):
        """
        tear down
        """
        log.info(("*" * 35) + " teardown " + ("*" * 35))
        log.info("[teardown_method] Start teardown test case %s..." %
                 method.__name__)
        log.info("skip drop collection")

    @pytest.fixture(scope="function", autouse=True)
    def connection(self, host, port):
        """
        connect milvus for each test case
        """
        connections.connect('default', host=host, port=port)
        if connections.has_connection("default") is False:
            raise Exception("no connections")
        log.info("connect to milvus successfully")

    def test_prepare(self):
        collection_w = ApiCollectionWrapper(active_trace=True)

        # check list collection
        collections, _ = self.utility_wrap.list_collections()
        if coll_name in collections:
            assert len(collections) == 1
            collection_w.init_collection(name=coll_name)
            # partitions = collection_w.partitions
            # assert len(partitions) == partition_num+1
        else:
            fields = [cf.gen_int64_field(), cf.gen_float_vec_field(dim=dim)]
            schema = cf.gen_collection_schema(fields, primary_field=ct.default_int64_field_name, auto_id=True)
            collection_w.init_collection(name=coll_name, schema=schema)

            # create many partitions
            for i in range(partition_num):
                partition_name = f"p_{i}"
                collection_w.create_partition(partition_name)

            partitions = collection_w.partitions
            log.info(f"collection {coll_name} has {len(partitions)} partitions")

    def test_insert(self):
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=coll_name)

        nb = 2000
        task_num = 100
        insert_loop = 200

        def do_insert():
            """
            do insert
            """
            vectors = cf.gen_vectors(nb, dim=dim)

            # loop 200 times, each time insert nb vectors into each partition
            for loop in range(insert_loop):
                for p in range(partition_num):
                    # random_p = random.randint(0, partition_num - 1)
                    _, res = collection_w.insert(data=[vectors], partition_name=f"p_{p}")
                    assert res

        with ThreadPoolExecutor(max_workers=100) as t:
            # submit 100 tasks
            all_tasks = [t.submit(do_insert) for _ in range(task_num)]
            wait(all_tasks, return_when=ALL_COMPLETED)
            log.info('finished all insert')
