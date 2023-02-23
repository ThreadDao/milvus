import random
import threading

import pytest
from pymilvus import connections

from base.collection_wrapper import ApiCollectionWrapper
from common import common_func as cf
from common import common_type as ct

from base.client_base import TestcaseBase
from utils.util_log import test_log as log

coll_name = "test_walmart"
partition_num = 200


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
        collection_w = ApiCollectionWrapper()

        # check list collection
        collections, _ = self.utility_wrap.list_collections()
        assert len(collections) == 1
        if coll_name in collections:
            collection_w.init_collection(name=coll_name)
            partitions = collection_w.partitions
            assert len(partitions) == partition_num
        else:
            fields = [cf.gen_int64_field(), cf.gen_float_vec_field()]
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

        nb = 5
        thread_num = 50
        threads = []

        def do_insert(thread_i):
            log.debug(f'In thread-{thread_i}')
            for loop in range(10000):
                random_p = random.randint(0, partition_num - 1)
                vectors = cf.gen_vectors(nb)
                _, res = collection_w.insert(data=vectors, partition_name=f"p_{random_p}")
                assert res

        for i in range(thread_num):
            x = threading.Thread(target=do_insert, args=(i,))
            threads.append(x)
            x.start()
        for t in threads:
            t.join()
