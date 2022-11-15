import time

import pytest
from pymilvus import connections, MilvusException

from base.collection_wrapper import ApiCollectionWrapper
from base.schema_wrapper import ApiCollectionSchemaWrapper
from base.utility_wrapper import ApiUtilityWrapper
from common import common_func as cf
from common import common_type as ct
from common.common_type import CheckTasks, CaseLabel
from utils.util_log import test_log as log


@pytest.mark.tags(CaseLabel.L3)
class TestMilvusChaos:
    """
    Base milvus operator collection for test milvus chaos memory
    """
    # collection and utility wrapper
    collection_w = ApiCollectionWrapper()
    utility_w = ApiUtilityWrapper()

    # float vector dimension
    dim = 512

    # init collection insert entities
    init_nb = 500000
    # init_nb = 10000

    # insert entities per time
    ni = 50000
    # ni = 1000

    @pytest.fixture(scope="function", autouse=True)
    def connect(self, host, port):
        """ connect """
        connections.connect("default", host=host, port=port)

    def test_init_collection(self, host, port, collection_name):
        """
        connect milvus and create collection and insert ?? entities 50w?
        """
        # create collection
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field(dim=self.dim)]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True,
                                                                        primary_field=ct.default_int64_field_name)
        self.collection_w.init_collection(collection_name, schema=schema)

        # insert init_nb entites
        for i in range(self.init_nb // self.ni):
            float_vec_values = cf.gen_vectors(self.ni, self.dim)
            insert_res, _ = self.collection_w.insert([float_vec_values], timeout=60)
            assert insert_res.succ_count == self.ni

            # flush data
            log.debug(f'collection num_entites after init: {self.collection_w.num_entities}')

    def test_insert(self, collection_name):
        """
        insert entities into init collection
        """
        duration = 300  # 5m
        self.collection_w.init_collection(collection_name)
        try:
            start = time.time()
            while time.time() - start < duration:
                float_vec_values = cf.gen_vectors(self.ni, self.dim)
                insert_res, _ = self.collection_w.insert([float_vec_values], timeout=40)
        except Exception as e:
            log.error(str(e))

    def test_create_index(self, collection_name):
        """
        create index
        """
        try:
            self.collection_w.init_collection(collection_name)
            index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 128}}
            self.collection_w.create_index(field_name=ct.default_float_vec_field_name,
                                           index_params=index_params, timeout=120)
        except MilvusException as e:
            log.error(str(e))
        finally:
            log.debug(f'Has index: {self.collection_w.has_index()[0]}')
            if self.collection_w.has_index()[0]:
                log.info(self.collection_w.indexes[0].params)

    def test_load_failed(self, collection_name, replica_num):
        """
        load collection and expected memory ecxeption
        """
        try:
            self.collection_w.init_collection(collection_name)
            index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 128}}
            self.collection_w.create_index(field_name=ct.default_float_vec_field_name,
                                           index_params=index_params, timeout=120)
            self.collection_w.load(replica_number=replica_num, check_task=CheckTasks.check_nothing)
            replicas, _ = self.collection_w.get_replicas()
            log.info(replicas)
            # todo sometimes get replicas successfully
            # self.collection_w.get_replicas(check_task=CheckTasks.err_res,
            #                                check_items={'err_code': 15,
            #                                             'err_msg': "collection not found, maybe not loaded"})
            segments_info, _ = self.utility_w.get_query_segment_info(collection_name)
            log.debug(f'{collection_name} querynode segments info {segments_info}')
        except MilvusException as e:
            log.error(str(e))

    def test_init_collection_check(self, collection_name, replica_num):
        """
        do some milvus operation to check init-collection
        """
        search_duration = 600
        # flush
        self.collection_w.init_collection(collection_name)
        log.debug(f'Start flush {collection_name}')
        log.debug(f'{collection_name} num entities is {self.collection_w.num_entities}')

        # load
        self.collection_w.load(replica_number=replica_num, timeout=120)
        log.debug(self.collection_w.get_replicas()[0])

        # new insert
        float_vec_values = cf.gen_vectors(self.ni, self.dim)
        insert_res, _ = self.collection_w.insert([float_vec_values], timeout=40)

        # search with SQ8 index
        search_res, _ = self.collection_w.search(cf.gen_vectors(ct.default_nq, dim=self.dim),
                                                 ct.default_float_vec_field_name, ct.default_search_params,
                                                 ct.default_limit, timeout=60)
        assert len(search_res[0]) == ct.default_limit

        # re-create index
        if self.collection_w.has_index()[0]:
            log.info(self.collection_w.indexes[0].params)
            self.collection_w.drop_index()
        index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 48, "efConstruction": 50}}
        search_params = {"metric_type": "L2", "params": {"ef": 50}}
        t0 = time.time()
        self.collection_w.create_index(field_name=ct.default_float_vec_field_name, index_params=index_params, timeout=1200)
        log.info(f"create index cost: {time.time() - t0}")
        assert self.collection_w.indexes[0].params == index_params
        self.collection_w.release()
        try:
            self.collection_w.load(replica_number=replica_num, timeout=120)
            start = time.time()
            while time.time() - start < search_duration:
                # search
                search_res, _ = self.collection_w.search(cf.gen_vectors(ct.default_nq, dim=self.dim),
                                                         ct.default_float_vec_field_name, search_params,
                                                         ct.default_limit, timeout=60)
                assert len(search_res[0]) == ct.default_limit

                # query
                query_res, _ = self.collection_w.query(f"{ct.default_int64_field_name} in {search_res[0].ids}")
                assert len(query_res) == ct.default_limit

                delete_expr = f"{ct.default_int64_field_name} in {search_res[-1].ids}"
                self.collection_w.delete(delete_expr)
                self.collection_w.query(delete_expr, check_task=CheckTasks.check_query_empty)
        except MilvusException as e:
            raise MilvusException(e)

        self.collection_w.compact()
        self.collection_w.wait_for_compaction_completed(timeout=3600)
        self.collection_w.get_compaction_plans(timeout=3600)