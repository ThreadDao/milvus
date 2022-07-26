import threading
from time import sleep

import pytest
import datetime

from pymilvus import connections
from pymilvus.exceptions import MilvusException

from base.collection_wrapper import ApiCollectionWrapper
from base.utility_wrapper import ApiUtilityWrapper
from common.cus_resource_opts import CustomResourceOperations as CusResource
from common import common_func as cf
from common import common_type as ct
from chaos.chaos_commons import gen_experiment_config
from common.common_type import CaseLabel, CheckTasks
from chaos import constants
from utils.util_log import test_log as log
from utils.util_k8s import wait_pods_ready


@pytest.mark.tags(CaseLabel.L3)
class TestDebugMemoryStress:
    nb = 100000
    dim = 128
    ni = 10000

    @pytest.fixture(scope="function", autouse=True)
    def prepare_collection(self, host, port):
        """ dim 128, 1000,000 entities loaded needed memory 3-5 Gi"""
        connections.connect("default", host=host, port=port)
        collection_w = ApiCollectionWrapper()
        c_name = "stress_replicas_2"
        collection_w.init_collection(name=c_name,
                                     schema=cf.gen_default_collection_schema(dim=self.dim))

        # insert 10 sealed segments
        for i in range(self.nb // self.ni):
            t0 = datetime.datetime.now()
            df = cf.gen_default_dataframe_data(nb=self.ni, dim=self.dim)
            res = collection_w.insert(df)[0]
            assert res.insert_count == self.ni
            log.info(f'After {i + 1} insert, num_entities: {collection_w.num_entities}')
            tt = datetime.datetime.now() - t0
            log.info(f"{i} insert and flush data cost: {tt}")

        log.debug(collection_w.num_entities)
        return collection_w

    @pytest.mark.parametrize("mode", ["one"])
    def test_memory_stress_replicas_group_insufficient(self, prepare_collection, mode):
        """
        target: test apply stress memory on different number querynodes and the group failed to load,
                bacause of the memory is insufficient
        method: 1.Limit querynodes memory 5Gi
                2.Create collection and insert 1000,000 entities
                3.Apply memory stress on querynodes and it's memory is not enough to load replicas
        expected: Verify load raise exception, and after delete chaos, load and search successfully
        """
        collection_w = prepare_collection
        utility_w = ApiUtilityWrapper()
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_querynode_memory_stress.yaml")

        # Update config
        chaos_config['spec']['mode'] = mode
        chaos_config['spec']['stressors']['memory']['size'] = '5Gi'
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        # chaos_start = time.time()
        log.debug("chaos injected")
        sleep(10)

        try:
            # load failed
            err = {"err_code": 1, "err_msg": "shuffleSegmentsToQueryNodeV2: insufficient memory of available node"}
            collection_w.load(replica_number=5, timeout=60, check_task=CheckTasks.err_res, check_items=err)

            # query failed because not loaded
            err = {"err_code": 1, "err_msg": "not loaded into memory"}
            collection_w.query("int64 in [0]", check_task=CheckTasks.err_res, check_items=err)

            # delete chaos
            meta_name = chaos_config.get('metadata', None).get('name', None)
            chaos_res.delete(metadata_name=meta_name)
            sleep(10)

            # after delete chaos load and query successfully
            collection_w.load(replica_number=5, timeout=60)
            progress, _ = utility_w.loading_progress(collection_w.name)
            # assert progress["loading_progress"] == "100%"
            query_res, _ = collection_w.query("int64 in [0]")
            assert len(query_res) != 0

            collection_w.release()

        except Exception as e:
            raise Exception(str(e))

        finally:
            log.debug("Test finished")


class TestIssue:
    def test_memory_stress_issue(self):
        connections.connect("default", host="10.98.0.9", port=19530)
        nb = 25000
        dim = 512
        ut = ApiUtilityWrapper()
        collection_w = ApiCollectionWrapper()
        c_name = "stress_replicas_2"
        collection_w.init_collection(name=c_name)

        # insert 10 sealed segments
        # for i in range(10):
        #     t0 = datetime.datetime.now()
        #     df = cf.gen_default_dataframe_data(nb=nb, dim=dim)
        #     res = collection_w.insert(df)[0]
        #     assert res.insert_count == nb
        #     log.info(f'After {i + 1} insert, num_entities: {collection_w.num_entities}')
        #     tt = datetime.datetime.now() - t0
        #     log.info(f"{i} insert and flush data cost: {tt}")
        log.debug(collection_w.num_entities)
        collection_w.release()

        collection_w.load(replica_number=2)
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=dim),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=60)
        collection_w.query("int64 in [2]")
        log.debug(ut.loading_progress(collection_w.name))
        log.debug(collection_w.get_replicas()[0])
        log.debug(ut.get_query_segment_info(collection_w.name))

    def test_issue_17091(self):
        connections.connect("default", host="10.98.0.9", port=19530)
        nb = 5000
        collection_w = ApiCollectionWrapper()
        ut = ApiUtilityWrapper()
        c_name = "stress_issue_17091"
        collection_w.init_collection(name=c_name,
                                     schema=cf.gen_default_collection_schema())
        # df = cf.gen_default_dataframe_data(nb=nb)
        # collection_w.insert(df)[0]
        log.debug(collection_w.num_entities)

        collection_w.release()
        collection_w.load(replica_number=2)
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=ct.default_dim),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=60)
        collection_w.query("int64 in [2]")
        ut.get_query_segment_info(collection_w.name)
        collection_w.get_replicas()

    def test_debug(self):
        connections.connect("default", host='10.98.0.4', port=19530)
        utility_w = ApiUtilityWrapper()
        collection_w = ApiCollectionWrapper()
        # utility_w.loading_progress('stress_replicas_2')
        collection_w.init_collection('stress_replicas_2')
        collection_w.release()
        collection_w.load(replica_number=2)
        for i in range(2):
            search_res, _ = collection_w.search(cf.gen_vectors(1, dim=512), ct.default_float_vec_field_name,
                                                ct.default_search_params, ct.default_limit, timeout=60)

    def test_chaos_memory_stress_insert_standalone(self, host):
        connections.connect("default", host=host, port=19530)
        collection_w = ApiCollectionWrapper()
        c_name = "stress_standalone"
        collection_w.init_collection(name=c_name,
                                     schema=cf.gen_default_collection_schema())

        nb = 1000

        def do_insert():
            """ do search """
            df = cf.gen_default_dataframe_data(nb=nb)
            res, is_succ = collection_w.insert(df, timeout=60, check_items=CheckTasks.check_nothing)
            assert res.insert_count == nb
            return res, is_succ

        def loop_insert():
            """ continuously search """
            for i in range(200):
                do_insert()

        try:
            t_insert = threading.Thread(target=loop_insert, args=())
            t_insert.start()

            chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_standalone_memory_stress.yaml")

            # Update config
            log.debug(chaos_config)
            chaos_res = CusResource(kind=chaos_config['kind'],
                                    group=constants.CHAOS_GROUP,
                                    version=constants.CHAOS_VERSION,
                                    namespace=constants.CHAOS_NAMESPACE)
            chaos_res.create(chaos_config)
            log.debug("chaos injected")

            t_insert.join()
        except MilvusException as e:
            wait_pods_ready("chaos-testing", "app.kubernetes.io/instance=milvus-chaos")
            do_insert()
        finally:
            meta_name = chaos_config.get('metadata', None).get('name', None)
            chaos_res.delete(metadata_name=meta_name)

        # log.debug(collection_w.num_entities)
        collection_w.load()
        search_res, _ = collection_w.search(cf.gen_vectors(ct.default_nq, ct.default_dim),
                                            ct.default_float_vec_field_name,
                                            ct.default_search_params, ct.default_limit)

    def test_tmp(self, host):
        import pdb
        connections.connect("default", host=host, port=19530)
        collection_w = ApiCollectionWrapper()
        c_name = "stress_standalone"
        collection_w.init_collection(name=c_name)
        # while True:
        #     df = cf.gen_default_dataframe_data(nb=1000)
        #     insert_res, _ = collection_w.insert(df, timeout=60, check_items=CheckTasks.check_nothing)
        #     # pdb.set_trace()
        #     if insert_res.insert_count == insert_res._mr.succ_count:
        #         break
        log.debug(collection_w.num_entities)
        collection_w.load()
        search_res, _ = collection_w.search(cf.gen_vectors(ct.default_nq, ct.default_dim),
                                            ct.default_float_vec_field_name,
                                            ct.default_search_params, ct.default_limit)
        for hits in search_res:
            log.debug(hits.ids)

    def test_chaos_memory_stress_search_standalone(self, host):
        connections.connect("default", host=host, port=19530)
        collection_w = ApiCollectionWrapper()
        dim = 128
        collection_w.init_collection(name='standalone_search_zNvU8YvJ',
                                     schema=cf.gen_default_collection_schema(dim=dim))

        nb = 50000
        ni = 5000
        # for i in range(nb // ni):
        #     df = cf.gen_default_dataframe_data(nb=ni, dim=dim, start=ni * i)
        #     insert_res, _ = collection_w.insert(df, timeout=60)

        log.debug(collection_w.num_entities)

        def do_search():
            """ do search """
            search_res, is_succ = collection_w.search(cf.gen_vectors(ct.default_nq, dim),
                                                      ct.default_float_vec_field_name,
                                                      ct.default_search_params, ct.default_limit)
            assert len(search_res) == ct.default_nq
            assert len(search_res[0]) == ct.default_limit
            return search_res, is_succ

        def loop_search(loop):
            """ continuously search """
            for i in range(loop):
                do_search()

        try:
            chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_standalone_memory_stress.yaml")

            # Update config
            log.debug(chaos_config)
            chaos_res = CusResource(kind=chaos_config['kind'],
                                    group=constants.CHAOS_GROUP,
                                    version=constants.CHAOS_VERSION,
                                    namespace=constants.CHAOS_NAMESPACE)
            chaos_res.create(chaos_config)
            log.debug("chaos injected")
            sleep(10)
            collection_w.load()
            t_search = threading.Thread(target=loop_search, args=(200,))
            t_search.start()

            t_search.join()
        except MilvusException as e:
            log.error(str(e))
            wait_pods_ready("chaos-testing", "app.kubernetes.io/instance=milvus-memory")

        finally:
            # delete chaos
            meta_name = chaos_config.get('metadata', None).get('name', None)
            chaos_res.delete(metadata_name=meta_name)
