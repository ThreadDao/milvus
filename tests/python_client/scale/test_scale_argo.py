from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import pytest
from pymilvus import connections, MilvusException
from scale import constants

from base.collection_wrapper import ApiCollectionWrapper
from base.schema_wrapper import ApiCollectionSchemaWrapper
from base.utility_wrapper import ApiUtilityWrapper
from common import common_func as cf
from common import common_type as ct
from common.common_type import CheckTasks, CaseLabel
from utils.util_log import test_log as log
from common.milvus_sys import MilvusSys
from utils.util_k8s import get_release_by_service_host, get_pod_list

prefix = "oom"

default_index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 200}}
default_search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
hnsw_search_params = {"metric_type": "L2", "params": {"ef": 64}}


@pytest.mark.tags(CaseLabel.L3)
class TestMilvusScale:
    # collection and utility wrapper
    collection_w = ApiCollectionWrapper()
    utility_w = ApiUtilityWrapper()

    # deploy mode standalone or cluster
    deploy_mode = ""

    @pytest.fixture(scope="function", autouse=True)
    def connect(self, host, port):
        """ connect """
        connections.connect("default", host=host, port=port)
        self.deploy_mode = MilvusSys().deploy_mode

    def test_insert_oom(self, host):
        # collection fields and schema
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_double_field(), cf.gen_float_vec_field(dim=3840)]

        def do_collection():
            """ create collection, insert, load, search, query"""
            schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)
            collection_w = ApiCollectionWrapper()
            collection_w.init_collection(name=cf.gen_unique_str(prefix), schema=schema, shards_num=16)
            for i in range(10):
                df = pd.DataFrame({
                    ct.default_double_field_name: pd.Series(data=[np.double(i) for i in range(0, 2000)],
                                                            dtype="double"),
                    ct.default_float_vec_field_name: cf.gen_vectors(2000, dim=3840)
                })
                insert_res, _ = collection_w.insert(df, timeout=180, check_task=CheckTasks.check_nothing)
            # log.debug(collection_w.num_entities)

            collection_w.load(timeout=120)
            log.debug(collection_w.get_replicas(), check_task=CheckTasks.check_nothing)
            query_res, _ = collection_w.query(expr=f"{ct.default_int64_field_name} in {insert_res.primary_keys[0:10]}",
                                              timeout=120)
            assert len(query_res) == 10
            search_res, _ = collection_w.search(cf.gen_vectors(2, dim=3840), ct.default_float_vec_field_name,
                                                default_search_params, 10, timeout=120)
            assert len(search_res) == 2
            assert len(search_res[0]) == 10

        try:
            tasks = []
            with ThreadPoolExecutor(max_workers=8) as t:
                for i in range(20):
                    task = t.submit(do_collection)
                    tasks.append(task)

            for task in tasks:
                task.done()
        except Exception as e:
            log.error(str(e))
            # get release by host ip
            service_component = "standalone" if self.deploy_mode == constants.STANDALONE_MODE else "proxy"
            status_component = "standalone" if self.deploy_mode == constants.STANDALONE_MODE else "datanode"
            release = get_release_by_service_host(host, component=service_component)

            # get pod container datanode or standalone status
            label = f"app.kubernetes.io/component={status_component},app.kubernetes.io/instance={release}"
            for item in get_pod_list(constants.NAMESPACE, label_selector=label):
                log.debug(f'pod name: {item.metadata.name}')
                status = item.status
                log.debug(f'container restart count:{status.container_statuses[0].restart_count}')
                log.debug(f'container status: {status.container_statuses[0].state}')
                log.debug(f'container last state: {status.container_statuses[0].last_state}')

    def test_oom_collection_check(self):
        """ check oom collection flush"""
        collections, _ = self.utility_w.list_collections()
        log.debug(f'collections: {collections}')
        for c in collections:
            if c.startswith(prefix):
                self.collection_w.init_collection(name=c)
                log.debug(f"{c} num entities: {self.collection_w.num_entities}")

    def test_index_oom(self):
        nb = 10000000  # 2m
        ni = 50000
        dim = 512
        # collection fields and schema
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field(dim=dim)]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)

        # init collection
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=cf.gen_unique_str(prefix), schema=schema)

        # insert data
        for i in range(nb // ni):
            df = pd.DataFrame({
                ct.default_float_vec_field_name: cf.gen_vectors(ni, dim=dim)
            })
            collection_w.insert(df, timeout=60)

        # flush data
        log.debug(f"Flush data num entities: {collection_w.num_entities}")

        # create index
        try:
            collection_w.create_index(ct.default_float_vec_field_name, default_index_params, timeout=1800)
        except Exception as e:
            log.error(str(e))

    def test_index_oom_collection_check(self):
        """
        check init collection
        """
        collection_w = ApiCollectionWrapper()
        collections, _ = self.utility_w.list_collections()
        log.debug(f'collections: {collections}')
        c_name = ""
        for c in collections:
            if c.startswith(prefix):
                c_name = c
        if c_name != "" and c_name.startswith(prefix):
            collection_w.init_collection(name=c_name)

            # create index, if oom actually indexnode is building the index
            if collection_w.has_index()[0]:
                index, _ = collection_w.index()
                log.debug(index.params == default_index_params)

            collection_w.create_index(ct.default_float_vec_field_name, default_index_params, timeout=18000)
            assert default_index_params == collection_w.index()[0].params
            collection_w.load(timeout=120)
            search_res, _ = collection_w.search(cf.gen_vectors(ct.default_nq, dim=512), ct.default_float_vec_field_name,
                                                hnsw_search_params, 10, timeout=120)
            assert len(search_res) == ct.default_nq
            assert len(search_res[0]) == ct.default_limit

        else:
            raise MilvusException(code=0, message="Failed to find index oom collection")

    def test_debug_index_oom(self):
        nb = 1000000  # 100w
        ni = 50000
        dim = 512
        # collection fields and schema
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field(dim=dim)]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)

        # init collection
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=cf.gen_unique_str(prefix), schema=schema, shards_num=1)

        # insert data
        for i in range(nb // ni):
            df = pd.DataFrame({
                ct.default_float_vec_field_name: cf.gen_vectors(ni, dim=dim)
            })
            collection_w.insert(df, timeout=60)

        # flush data
        log.debug(f"Flush data num entities: {collection_w.num_entities}")

        # create index
        try:
            collection_w.create_index(ct.default_float_vec_field_name, default_index_params, timeout=1800)
        except Exception as e:
            log.error(str(e))

    def test_load_oom(self):
        nb = 550000  # 2m
        ni = 50000
        dim = 100

        # collection fields and schema
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field(dim=dim)]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)

        # init collection
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=cf.gen_unique_str(prefix), schema=schema)

        # insert data and flush
        for i in range(nb // ni):
            df = pd.DataFrame({
                ct.default_float_vec_field_name: cf.gen_vectors(ni, dim=dim)
            })
            collection_w.insert(df)

        log.debug(f'num entities: {collection_w.num_entities}')

        # create index
        # index_params = {
        #     "index_type": "IVF_FLAT",
        #     "metric_type": "L2",
        #     "params": {"nlist": 4096},
        # }
        collection_w.create_index(ct.default_float_vec_field_name, default_index_params, timeout=1800)

        # load and search
        try:
            collection_w.load(timeout=300)
            # search_params = {
            #     "metric_type": "L2",
            #     "params": {"nprobe": 128},
            # }
            search_res, _ = collection_w.search(cf.gen_vectors(3, dim=dim), ct.default_float_vec_field_name,
                                                hnsw_search_params, 5, consistency_level="Eventually", timeout=1200)
            assert len(search_res) == 3
            assert len(search_res[0]) == 5
        except Exception as e:
            log.error(str(e))

    def test_load_oom_collection_check(self):
        collection_w = ApiCollectionWrapper()
        utility_w = ApiUtilityWrapper()
        dim = 100

        # find colelction
        collections, _ = self.utility_w.list_collections()
        log.debug(f'collections: {collections}')
        c_name = ""
        for c in collections:
            if c.startswith(prefix):
                c_name = c

        if c != "" and c_name.startswith(prefix):
            collection_w.init_collection(name=c_name)

            # querynode info
            collection_w.load()
            replicas, _ = collection_w.get_replicas(timeout=60)
            log.debug(replicas)

            seg_info, _ = utility_w.get_query_segment_info(collection_w.name)
            log.debug(seg_info)

            # search
            search_params = {"metric_type": "L2", "params": {"nprobe": 128}}
            search_res, _ = collection_w.search(cf.gen_vectors(3, dim=dim), ct.default_float_vec_field_name,
                                                hnsw_search_params, 5, timeout=1200)
            assert len(search_res) == 3
            assert len(search_res[0]) == 5

        else:
            raise MilvusException(code=0, message="Failed to find load oom collection")
    
    def test_init_scale_collection(self):
        nb = 5000000  # 100w
        ni = 50000
        dim = 512
        # collection fields and schema
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field(dim=dim)]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)

        # init collection
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=cf.gen_unique_str(prefix), schema=schema)

        # insert data
        for i in range(nb // ni):
            df = pd.DataFrame({
                ct.default_float_vec_field_name: cf.gen_vectors(ni, dim=dim)
            })
            collection_w.insert(df, timeout=60)

        # flush data
        log.debug(f"Flush data num entities: {collection_w.num_entities}")