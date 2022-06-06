import random
import threading
import multiprocessing
import time
from time import sleep

import pytest
import datetime

from pymilvus import connections
from base.collection_wrapper import ApiCollectionWrapper
from base.utility_wrapper import ApiUtilityWrapper
from chaos.checker import Op, CreateChecker, InsertFlushChecker, IndexChecker, SearchChecker, QueryChecker
from common.cus_resource_opts import CustomResourceOperations as CusResource
from common import common_func as cf
from common import common_type as ct
from chaos import chaos_commons as cc
from chaos.chaos_commons import gen_experiment_config, get_chaos_yamls, start_monitor_threads
from common.common_type import CaseLabel, CheckTasks
from chaos import constants
from utils.util_log import test_log as log
from utils.util_k8s import get_querynode_id_pod_pairs


def apply_memory_stress(chaos_yaml):
    chaos_config = gen_experiment_config(chaos_yaml)
    log.debug(chaos_config)
    chaos_res = CusResource(kind=chaos_config['kind'],
                            group=constants.CHAOS_GROUP,
                            version=constants.CHAOS_VERSION,
                            namespace=constants.CHAOS_NAMESPACE)
    chaos_res.create(chaos_config)
    log.debug("chaos injected")


@pytest.mark.tags(CaseLabel.L3)
class TestChaosData:

    @pytest.fixture(scope="function", autouse=True)
    def connection(self, host, port):
        connections.add_connection(default={"host": host, "port": port})
        connections.connect(alias='default')

    @pytest.mark.tags(CaseLabel.L3)
    def test_memory_stress_replicas_before_load(self):
        """
        target: test querynode group load with insufficient memory
        method: 1.Limit querynode memory 2Gi
                2.Insert 50w 512 dim data
                2.Load sealed data (needed memory > memory limit) and search
        expected: Loading without exception but actually failed;
                  get_replicas and loading_progress get not loaded exception;
                  get_query_segment_info return empty list
                  querynode maybe OOMKilled, but can be restarted
        """
        host = '10.98.0.4'
        nb = 500000
        dim = 512
        ni = 10000
        connections.connect(host=host, port=19530)
        collection_w = ApiCollectionWrapper()
        utility_w = ApiUtilityWrapper()
        collection_w.init_collection(name=cf.gen_unique_str('before_load'),
                                     schema=cf.gen_default_collection_schema(dim=dim))

        for i in range(nb // ni):
            t0 = datetime.datetime.now()
            df = cf.gen_default_dataframe_data(nb=ni, dim=dim, start=i * ni)
            res = collection_w.insert(df, timeout=60)[0]
            assert res.insert_count == ni
        log.info(f'num_entities: {collection_w.num_entities}')
        tt = datetime.datetime.now() - t0
        log.info(f"cost: {tt}")

        collection_w.load(replica_number=2)
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=dim), ct.default_float_vec_field_name,
                                            ct.default_search_params, ct.default_limit, timeout=60,
                                            check_task=CheckTasks.err_res,
                                            check_items={"err_code": 1, "err_msg": "fail to Search,"})
        collection_w.get_replicas(check_task=CheckTasks.err_res,
                                  check_items={"err_code": 15,
                                               "err_msg": "collection not found, maybe not loaded"})
        utility_w.loading_progress(collection_w.name, check_task=CheckTasks.err_res,
                                   check_items={"err_code": 1,
                                                "err_msg": "has not been loaded into QueryNode"})
        # {'loading_progress': '0%', 'num_loaded_partitions': 0, 'not_loaded_partitions': ['_default']}
        seg_info, _ = utility_w.get_query_segment_info(collection_w.name)
        assert len(seg_info) == 0

    def test_chaos_memory_standalone(self):
        collection_w = ApiCollectionWrapper()
        utility_w = ApiUtilityWrapper()
        collection_w.init_collection(cf.gen_unique_str('before_insert'), cf.gen_default_collection_schema(dim=512),
                                     shards_num=4)
        for _ in range(10):
            df = cf.gen_default_dataframe_data(nb=10000, dim=512)
            collection_w.insert(df, timeout=60)

        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_standalone_memory_stress.yaml")
        chaos_config['spec']['duration'] = '5m'
        chaos_config['spec']['stressors']['memory']['size'] = '6Gi'
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.debug("chaos injected")
        sleep(15)

        try:
            index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 128}}
            t0_index = datetime.datetime.now()
            index, _ = collection_w.create_index(field_name=ct.default_float_vec_field_name,
                                                 index_params=index_params)
            tt_index = datetime.datetime.now() - t0_index

            log.info(f"create index cost: {tt_index}")
            log.info(collection_w.indexes[0].params)
            collection_w.load()
            utility_w.loading_progress(collection_w.name)
            collection_w.get_replicas()
            utility_w.get_query_segment_info(collection_w.name)
            search_res, _ = collection_w.search(cf.gen_vectors(1, dim=512), ct.default_float_vec_field_name,
                                                ct.default_search_params, ct.default_limit, timeout=60)

            collection_w.query(f'{ct.default_float_vec_field_name} in [0, 100, 99, 767]')

        except Exception as e:
            log.error(str(e))

        finally:
            # delete chaos
            meta_name = chaos_config.get('metadata', None).get('name', None)
            chaos_res.delete(metadata_name=meta_name)

    @pytest.mark.tags(CaseLabel.L3)
    @pytest.mark.parametrize('chaos_yaml', get_chaos_yamls())
    def test_chaos_memory_stress_querynode(self, connection, chaos_yaml):
        """
        target: explore query node behavior after memory stress chaos injected and recovered
        method: 1. Create a collection, insert some data
                2. Inject memory stress chaos
                3. Start a thread to load, search and query
                4. After chaos duration, check query search success rate
                5. Delete chaos or chaos finished finally
        expected: 1.If memory is insufficient, querynode is OOMKilled and available after restart
                  2.If memory is sufficient, succ rate of query and search both are 1.0
        """
        c_name = 'chaos_memory_nx6DNW4q'
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(c_name)
        log.debug(collection_w.schema)
        log.debug(collection_w._shards_num)

        # apply memory stress chaos
        chaos_config = gen_experiment_config(chaos_yaml)
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.debug("chaos injected")
        duration = chaos_config.get('spec').get('duration')
        duration = duration.replace('h', '*3600+').replace('m', '*60+').replace('s', '*1+') + '+0'
        meta_name = chaos_config.get('metadata').get('name')
        # wait memory stress
        sleep(constants.WAIT_PER_OP * 2)

        # try to do release, load, query and serach in a duration time loop
        try:
            start = time.time()
            while time.time() - start < eval(duration):
                collection_w.release()
                collection_w.load()

                term_expr = f'{ct.default_int64_field_name} in {[random.randint(0, 100)]}'
                query_res, _ = collection_w.query(term_expr)
                assert len(query_res) == 1

                search_res, _ = collection_w.search(cf.gen_vectors(1, ct.default_dim),
                                                    ct.default_float_vec_field_name,
                                                    ct.default_search_params, ct.default_limit)
                log.debug(search_res[0].ids)
                assert len(search_res[0].ids) == ct.default_limit

        except Exception as e:
            raise Exception(str(e))

        finally:
            chaos_res.delete(meta_name)

    @pytest.mark.tags(CaseLabel.L3)
    def test_chaos_memory_stress_datanode(self):
        """
        target: test inject memory stress into dataNode
        method: 1.Deploy milvus and limit datanode memory resource
                2.Create collection and insert some data
                3.Inject memory stress chaos
                4.Continue to insert data
        expected:
        """
        # init collection and insert 250 nb
        nb = 500000
        dim = 512
        c_name = cf.gen_unique_str('chaos_memory')
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=c_name)
        # for i in range(10):
        #     t0 = datetime.datetime.now()
        #     df = cf.gen_default_dataframe_data(nb=nb, dim=dim)
        #     res = collection_w.insert(df)[0]
        #     assert res.insert_count == nb
        #     log.info(f'After {i + 1} insert, num_entities: {collection_w.num_entities}')
        #     tt = datetime.datetime.now() - t0
        #     log.info(f"{i} insert and flush data cost: {tt}")

        # inject memory stress
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_datanode_memory_stress.yaml")
        chaos_config['spec']['duration'] = '5m'
        chaos_config['spec']['stressors']['memory']['size'] = '3Gi'
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.debug("chaos injected")
        sleep(20)
        try:
            # Continue to insert data
            for i in range(2):
                df = cf.gen_default_dataframe_data(nb=10000, dim=512)
                collection_w.insert(df, check_task=CheckTasks.check_nothing, timeout=60)
            log.info(f'Total num entities: {collection_w.num_entities}')

        except Exception as e:
            log.error(str(e))

        finally:
            # delete chaos
            meta_name = chaos_config.get('metadata', None).get('name', None)
            chaos_res.delete(metadata_name=meta_name)

    @pytest.mark.tags(CaseLabel.L3)
    def test_chaos_memory_stress_indexnode(self, connection):
        """
        target: test inject memory stress into indexnode
        method: 1.Deploy milvus and limit indexnode memory resource 3 / 4Gi
                2.Create collection and insert some data
                3.Inject memory stress chaos 512Mi
                4.Create index
        expected:
        """
        # init collection and insert
        nb = 256000  # vector size: 512*4*nb about 512Mi and create index need 2.8Gi memory
        dim = 512
        # c_name = cf.gen_unique_str('chaos_memory')
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_indexnode_memory_stress.yaml")
        chaos_config['spec']['duration'] = '5m'
        chaos_config['spec']['stressors']['memory']['size'] = '6Gi'
        c_name = 'stress_replicas_Oh6xULTy'
        index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 128}}

        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=c_name)
        # collection_w.init_collection(name=c_name, schema=cf.gen_default_collection_schema(dim=dim), shards_num=1)

        # insert 256000 512 dim entities, size 512Mi
        for i in range(2):
            t0_insert = datetime.datetime.now()
            df = cf.gen_default_dataframe_data(nb=10000, dim=dim)
            collection_w.insert(df, timeout=60)
            # log.info(f'After {i + 1} insert, num_entities: {collection_w.num_entities}')
            tt_insert = datetime.datetime.now() - t0_insert
            log.info(f"{i} insert data cost: {tt_insert}")

        # flush
        # t0_flush = datetime.datetime.now()
        log.debug(collection_w.num_entities)
        # tt_flush = datetime.datetime.now() - t0_flush
        # log.info(f'flush {nb * 10} entities cost: {tt_flush}')

        # log.info(collection_w.indexes[0].params)
        if collection_w.has_index()[0]:
            collection_w.drop_index()

        # indexNode start build index, inject chaos memory stress
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.debug("inject chaos")
        sleep(20)

        try:
            # create index
            t0_index = datetime.datetime.now()
            index, _ = collection_w.create_index(field_name=ct.default_float_vec_field_name,
                                                 index_params=index_params)
            tt_index = datetime.datetime.now() - t0_index

            log.info(f"create index cost: {tt_index}")
            log.info(collection_w.indexes[0].params)
        except Exception as e:
            log.error(str(e))
        finally:
            meta_name = chaos_config.get('metadata', None).get('name', None)
            chaos_res.delete(metadata_name=meta_name)

    def test_index_insufficient_memory(self, connection):
        collection_w = ApiCollectionWrapper()
        c_name = 'stress_replicas_Oh6xULTy'
        index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 128}}
        collection_w.init_collection(name=c_name)

        # t0_index = datetime.datetime.now()
        # index, _ = collection_w.create_index(field_name=ct.default_float_vec_field_name,
        #                                      index_params=index_params)
        # tt_index = datetime.datetime.now() - t0_index
        #
        # log.info(f"create index cost: {tt_index}")
        log.info(collection_w.indexes[0].params)

    @pytest.mark.tags(CaseLabel.L3)
    def test_chaos_memory_stress_etcd(self):
        """
        target: test inject memory stress into all etcd pods
        method: 1.Deploy milvus and limit etcd memory resource 1Gi with all mode
                2.Continuously and concurrently do milvus operations
                3.Inject memory stress chaos 51024Mi
                4.After duration, delete chaos stress
        expected: Verify milvus operation succ rate
        """
        mic_checkers = {
            Op.create: CreateChecker(),
            Op.insert: InsertFlushChecker(),
            Op.flush: InsertFlushChecker(flush=True),
            Op.index: IndexChecker(),
            Op.search: SearchChecker(),
            Op.query: QueryChecker()
        }
        # start thread keep running milvus op
        start_monitor_threads(mic_checkers)

        # parse chaos object
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_indexnode_memory_stress.yaml")
        chaos_config['spec']['duration'] = '5m'  # duration = chaos_config["spec"]["duration"]
        meta_name = chaos_config.get('metadata').get('name')
        duration = chaos_config.get('spec').get('duration')

        # apply chaos object
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.info("Chaos injected")

        # convert string duration time to an int number in seconds
        if isinstance(duration, str):
            duration = duration.replace('h', '*3600+').replace('m', '*60+').replace('s', '*1+') + '+0'
        else:
            log.error("Duration must be string type")

        # Delete experiment after it's over
        timer = threading.Timer(interval=eval(duration), function=chaos_res.delete, args=(meta_name, False))
        timer.start()
        timer.join()

        # output milvus op succ rate
        for k, ch in mic_checkers.items():
            log.debug(f'Succ rate of {k.value}: {ch.succ_rate()}')
            assert ch.succ_rate() == 1.0


@pytest.mark.tags(CaseLabel.L3)
class TestMemoryStressReplica:

    # @pytest.fixture(scope="function", autouse=True)
    def init_chaos_collection(self, host, port):
        """ dim 128, 1000,000 entities loaded needed memory 3-5 Gi"""
        ni = 50000
        dim = 128
        nb = 1000000
        connections.connect(host=host, port=19530)
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name=cf.gen_unique_str('stress_replicas'),
                                     schema=cf.gen_default_collection_schema(dim=dim))

        t0 = datetime.datetime.now()
        for i in range(nb // ni):
            df = cf.gen_default_dataframe_data(nb=ni, dim=dim)
            res = collection_w.insert(df, timeout=60)[0]
            assert res.insert_count == ni
        log.info(f'num_entities: {collection_w.num_entities}')
        tt = datetime.datetime.now() - t0
        log.info(f"insert and flush data cost: {tt}")

        log.debug(collection_w.num_entities)
        return collection_w

    @pytest.mark.parametrize("mode", ["one", "all", "fixed"])
    @pytest.mark.tags(CaseLabel.L3)
    def test_memory_stress_replicas_group_sufficient(self, mode):
        """
        target: test apply stress memory on one querynode and the memory is enough to load replicas
        method: 1.Limit all querynode memory 6Gi
                2.Apply 3Gi memory stress on different number of querynodes (load whole collection need about 1.5GB)
        expected: Verify load successfully and search result are correct
        """

        collection_w = ApiCollectionWrapper()
        utility_w = ApiUtilityWrapper()
        connections.connect(host='10.98.0.4', port=19530)
        collection_w.init_collection(name='stress_replicas_iYtgS65s')

        # # apply memory stress chaos
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_querynode_memory_stress.yaml")
        chaos_config['spec']['mode'] = mode
        chaos_config['spec']['duration'] = '3m'
        chaos_config['spec']['stressors']['memory']['size'] = '3Gi'
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.debug("chaos injected")
        sleep(20)
        #
        try:
            collection_w.load(replica_number=2, timeout=60)
            utility_w.loading_progress(collection_w.name)
            replicas, _ = collection_w.get_replicas()
            log.debug(replicas)
            search_res, _ = collection_w.search(cf.gen_vectors(1, dim=ct.default_dim),
                                                ct.default_float_vec_field_name, ct.default_search_params,
                                                ct.default_limit, timeout=120)
            assert 1 == len(search_res) and ct.default_limit == len(search_res[0])
            collection_w.release()

        except Exception as e:
            raise Exception(str(e))

        finally:
            # delete chaos
            meta_name = chaos_config.get('metadata', None).get('name', None)
            chaos_res.delete(metadata_name=meta_name)
            sleep(5)
            log.debug("Test finished")

    @pytest.mark.parametrize("mode", ["one", "all", "fixed"])
    def test_memory_stress_replicas_group_insufficient(self, mode):
        """
        target: test apply stress memory on different number querynodes and the group failed to load,
                because of the memory is insufficient
        method: 1.Limit querynode memory 6Gi
                2.Create collection and insert 1000,000 entities
                3.Apply memory stress on querynodes and it's memory is not enough to load replicas
        expected: Verify load raise exception, and after delete chaos, load and search successfully
        """
        collection_w = ApiCollectionWrapper()
        utility_w = ApiUtilityWrapper()
        connections.connect(host='10.98.0.4', port=19530)
        collection_w.init_collection(name='stress_replicas_iYtgS65s')
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_querynode_memory_stress.yaml")

        # Update config
        chaos_config['spec']['mode'] = mode
        chaos_config['spec']['stressors']['memory']['size'] = '6Gi'
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        # chaos_start = time.time()
        log.debug("chaos injected")
        sleep(20)

        try:
            # load failed
            # err = {"err_code": 1, "err_msg": "shuffleSegmentsToQueryNodeV2: insufficient memory of available node"}
            err = {'err_code': 1, 'err_msg': "no enough memory to load collection/partitions"}
            collection_w.load(replica_number=5, timeout=60, check_task=CheckTasks.err_res, check_items=err)

            # query failed because not loaded or checkIfLoaded failed when query
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
            # meta_name = chaos_config.get('metadata', None).get('name', None)
            # chaos_res.delete(metadata_name=meta_name)
            sleep(5)
            log.debug("Test finished")

    @pytest.mark.parametrize("mode", ["one", "all", "fixed"])
    def test_chaos_memory_stress_replicas_OOM(self, prepare_collection, mode):
        """
        target: test apply memory stress during loading, and querynode OOMKilled
        method: 1.Deploy and limit querynode memory limit 6Gi
                2.Create collection and insert 1000,000 entities
                3.Apply memory stress and querynode OOMKilled during loading replicas
        expected: Verify the mic is available to load and search querynode restart
        """
        collection_w = prepare_collection
        utility_w = ApiUtilityWrapper()

        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_querynode_memory_stress.yaml")
        chaos_config['spec']['mode'] = mode
        chaos_config['spec']['duration'] = '3m'
        chaos_config['spec']['stressors']['memory']['size'] = '6Gi'
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)

        chaos_res.create(chaos_config)
        log.debug("chaos injected")
        collection_w.load(replica_number=2, timeout=60, _async=True)

        utility_w.wait_for_loading_complete(collection_w.name)
        progress, _ = utility_w.loading_progress(collection_w.name)
        assert progress["loading_progress"] == '100%'

        sleep(180)
        chaos_res.delete(metadata_name=chaos_config.get('metadata', None).get('name', None))

        # TODO search failed
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=self.dim),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=120)
        assert 1 == len(search_res) and ct.default_limit == len(search_res[0])

        collection_w.release()
        collection_w.load(replica_number=2)
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=self.dim),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=120)
        assert 1 == len(search_res) and ct.default_limit == len(search_res[0])
        collection_w.release()


@pytest.mark.tags(CaseLabel.L3)
class TestMemoryStressReplicaLoadBalance:
    nb = 5000
    dim = 128

    @pytest.fixture(scope="function", autouse=True)
    def prepare_collection(self, host, port):
        """ dim 128, 1000,000 entities loaded needed memory 3-5 Gi"""
        connections.connect("default", host=host, port=19530)
        collection_w = ApiCollectionWrapper()
        c_name = "stress_replicas_1"
        collection_w.init_collection(name=c_name, shards_num=2,
                                     schema=cf.gen_default_collection_schema(dim=self.dim))

        collection_w.insert(cf.gen_default_dataframe_data(nb=self.nb, dim=self.dim))

        assert collection_w.num_entities == self.nb
        log.debug(collection_w.num_entities)
        return collection_w

    @pytest.mark.skip(reason="https://github.com/milvus-io/milvus/issues/17040")
    def test_memory_stress_replicas_group_load_balance(self, prepare_collection):
        """
        target: test apply memory stress on replicas and load balance inside group
        method: 1.Deploy milvus 3 querynodes and limit querynode memory 6Gi
                2.Insert 1000,000 entities (500Mb), load 2 replicas (memory usage 1.5Gb)
                3.Apply memory stress 4Gi on querynode
        expected: Verify that load balancing occurs
        """
        collection_w = prepare_collection
        utility_w = ApiUtilityWrapper()
        release_name = "mic-memory"

        # load and search
        collection_w.load(replica_number=2)
        progress, _ = utility_w.loading_progress(collection_w.name)
        assert progress["loading_progress"] == "100%"

        # get the replica and random chaos querynode
        replicas, _ = collection_w.get_replicas()
        chaos_querynode_id = replicas.groups[0].group_nodes[0]
        label = f"app.kubernetes.io/instance={release_name}, app.kubernetes.io/component=querynode"
        querynode_id_pod_pair = get_querynode_id_pod_pairs("chaos-testing", label)
        chaos_querynode_pod = querynode_id_pod_pair[chaos_querynode_id]

        # get the segment num before chaos
        seg_info_before, _ = utility_w.get_query_segment_info(collection_w.name)
        seg_distribution_before = cf.get_segment_distribution(seg_info_before)
        segments_num_before = len(seg_distribution_before[chaos_querynode_id]["sealed"])
        log.debug(segments_num_before)
        log.debug(seg_distribution_before[chaos_querynode_id]["sealed"])

        # apply memory stress
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_replicas_memory_stress_pods.yaml")
        chaos_config['spec']['selector']['pods']['chaos-testing'] = [chaos_querynode_pod]
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.debug(f"Apply memory stress on querynode {chaos_querynode_id}, pod {chaos_querynode_pod}")

        duration = chaos_config.get('spec').get('duration')
        duration = duration.replace('h', '*3600+').replace('m', '*60+').replace('s', '*1+') + '+0'
        sleep(eval(duration))

        chaos_res.delete(metadata_name=chaos_config.get('metadata', None).get('name', None))

        # Verfiy auto load loadbalance
        seg_info_after, _ = utility_w.get_query_segment_info(collection_w.name)
        seg_distribution_after = cf.get_segment_distribution(seg_info_after)
        segments_num_after = len(seg_distribution_after[chaos_querynode_id]["sealed"])
        log.debug(segments_num_after)
        log.debug(seg_distribution_after[chaos_querynode_id]["sealed"])

        assert segments_num_after < segments_num_before
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=self.dim),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=120)
        assert 1 == len(search_res) and ct.default_limit == len(search_res[0])

    def test_memory_stress_replicas_cross_group_load_balance(self):
        """
        target: test apply memory stress on one group and no load balance cross replica groups
        method: 1.Limit all querynodes memory 6Gi
                2.Create and insert 1000,000 entities
                3.Load collection with two replicas
                4.Apply memory stress on one grooup 80%
        expected: Verify that load balancing across groups is not occurring
        """
        collection_w = ApiCollectionWrapper()
        collection_w.init_collection(name='stress_replicas_Oh6xULTy')
        utility_w = ApiUtilityWrapper()
        release_name = "mic-zong"

        # load and searchc
        collection_w.load(replica_number=2)
        progress, _ = utility_w.loading_progress(collection_w.name)
        assert progress["loading_progress"] == "100%"
        seg_info_before, _ = utility_w.get_query_segment_info(collection_w.name)

        # get the replica and random chaos querynode
        replicas, _ = collection_w.get_replicas()
        group_nodes = list(replicas.groups[0].group_nodes)
        label = f"app.kubernetes.io/instance={release_name}, app.kubernetes.io/component=querynode"
        querynode_id_pod_pair = get_querynode_id_pod_pairs("chaos-testing", label)
        group_nodes_pod = [querynode_id_pod_pair[node_id] for node_id in group_nodes]

        # apply memory stress
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_replicas_memory_stress_pods.yaml")
        chaos_config['spec']['selector']['pods']['chaos-testing'] = group_nodes_pod
        log.debug(chaos_config)
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)
        log.debug(f"Apply memory stress on querynode {group_nodes}, pod {group_nodes_pod}")

        duration = chaos_config.get('spec').get('duration')
        duration = duration.replace('h', '*3600+').replace('m', '*60+').replace('s', '*1+') + '+0'
        sleep(eval(duration))

        chaos_res.delete(metadata_name=chaos_config.get('metadata', None).get('name', None))

        # Verfiy auto load loadbalance
        seg_info_after, _ = utility_w.get_query_segment_info(collection_w.name)
        seg_distribution_before = cf.get_segment_distribution(seg_info_before)
        seg_distribution_after = cf.get_segment_distribution(seg_info_after)
        for node_id in group_nodes:
            assert len(seg_distribution_before[node_id]) == len(seg_distribution_after[node_id])

        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=512),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=120)
        assert 1 == len(search_res) and ct.default_limit == len(search_res[0])

    @pytest.mark.tags(CaseLabel.L3)
    def test_memory_stress_replicas_load_balance_single_node(self, prepare_collection):
        """
        target: test apply memory stress on single node replica, and it OOMKilled
        method: 1.Deploy 2 querynodes and limit memory 6Gi
                2.Loading 1000,000 entities (data_size=500Mb) with 2 replicas (memory_usage=1.5Gb)
                3.Apply memory stress on one querynode and make it OOMKilled
        expected: After deleting chaos, querynode turns running, search successfully
        """
        collection_w = prepare_collection
        utility_w = ApiUtilityWrapper()

        # load and search
        collection_w.load(replica_number=2)
        progress, _ = utility_w.loading_progress(collection_w.name)
        assert progress["loading_progress"] == "100%"
        query_res, _ = collection_w.query("int64 in [0]")
        assert len(query_res) != 0

        # apply memory stress
        chaos_config = gen_experiment_config("./chaos_objects/memory_stress/chaos_querynode_memory_stress.yaml")

        # Update config
        chaos_config['spec']['mode'] = "one"
        chaos_config['spec']['stressors']['memory']['size'] = '6Gi'
        chaos_config['spec']['duration'] = "4m"
        log.debug(chaos_config)
        duration = chaos_config.get('spec').get('duration')
        duration = duration.replace('h', '*3600+').replace('m', '*60+').replace('s', '*1+') + '+0'
        chaos_res = CusResource(kind=chaos_config['kind'],
                                group=constants.CHAOS_GROUP,
                                version=constants.CHAOS_VERSION,
                                namespace=constants.CHAOS_NAMESPACE)
        chaos_res.create(chaos_config)

        sleep(eval(duration))
        chaos_res.delete(metadata_name=chaos_config.get('metadata', None).get('name', None))

        # release and load again
        collection_w.release()
        collection_w.load(replica_number=2)
        progress, _ = utility_w.loading_progress(collection_w.name)
        assert progress["loading_progress"] == "100%"
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=self.dim),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=120)
        assert 1 == len(search_res) and ct.default_limit == len(search_res[0])

    def test_load_balance(self, prepare_collection):
        """
        target: test apply memory stress on replicas and load balance inside group
        method: 1.Deploy milvus and limit querynode memory 6Gi
                2.Insret 1000,000 entities (500Mb), load 2 replicas (memory usage 1.5Gb)
                3.Apply memory stress 4Gi on querynode
        expected: Verify that load balancing occurs
        """
        collection_w = prepare_collection
        utility_w = ApiUtilityWrapper()
        release_name = "mic-replica"

        # load and searchc
        collection_w.load(replica_number=2)
        progress, _ = utility_w.loading_progress(collection_w.name)
        assert progress["loading_progress"] == "100%"

        # get the replica and random chaos querynode
        # seg_info, _ = utility_w.get_query_segment_info(collection_w.name)
        # log.debug(seg_info)
        search_res, _ = collection_w.search(cf.gen_vectors(1, dim=self.dim),
                                            ct.default_float_vec_field_name, ct.default_search_params,
                                            ct.default_limit, timeout=120)
        assert 1 == len(search_res) and ct.default_limit == len(search_res[0])
