import random

import pytest
from pymilvus import connections

from base.collection_wrapper import ApiCollectionWrapper
from base.utility_wrapper import ApiUtilityWrapper
from base.schema_wrapper import ApiCollectionSchemaWrapper
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel, CheckTasks
from utils.util_log import test_log as log

default_index_params = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 64}}
nb = 20000
ni = 2000


@pytest.mark.tags(CaseLabel.L3)
class TestUpgradeIndex:
    collection_name = "upgrade_index"
    collection_w = ApiCollectionWrapper()
    utility_w = ApiUtilityWrapper()

    @pytest.fixture(scope="function", autouse=True)
    def test_connect(self, host, port):
        connections.connect("default", host=host, port=port)

    def test_index_before_upgrade(self):
        """
        target: test build index with old image, handoff, search with new image
        method: before upgrade: init -> insert-flush -> index
        expected: all succ
        """
        # init collection
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field()]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)
        self.collection_w.init_collection(name=self.collection_name, schema=schema)

        # insert data
        for i in range(nb // ni):
            data = [cf.gen_vectors(nb=ni, dim=ct.default_dim)]
            self.collection_w.insert(data)
            log.debug(f"num entities: {self.collection_w.collection.num_entities}")

        # create index
        self.collection_w.create_index(ct.default_float_vec_field_name, default_index_params,
                                       index_name=ct.default_index_name, timeout=3600)
        assert self.collection_w.indexes[0].params == default_index_params
        log.debug(f"collection index: {self.collection_w.indexes[0].params}")

        self.utility_w.list_collections()

    def test_check_index_before_upgrade(self):
        """
        target: test build index with old image, handoff, search with new image
        method: after upgrade: search -> insert-handoff -> delete-query -> compact -> re-index -> search
        expected: all succ
        """
        # describe index
        self.utility_w.list_collections()
        self.collection_w.init_collection(name=self.collection_name)
        log.debug(self.collection_w.num_entities)
        if self.collection_w.has_index()[0]:
            log.debug(self.collection_w.indexes[0].params)
        assert self.collection_w.indexes[0].index_name == ct.default_index_name

        # search succ
        self.collection_w.load(timeout=20)
        query_vectors = [[random.random() for _ in range(ct.default_dim)] for _ in range(ct.default_nq)]
        search_res, _ = self.collection_w.search(query_vectors,
                                                 ct.default_float_vec_field_name,
                                                 ct.default_search_params, ct.default_limit)
        assert len(search_res) == ct.default_nq
        assert len(search_res[0]) == ct.default_limit

        # new insert, flush and handoff
        insert_res, _ = self.collection_w.insert([cf.gen_vectors(nb=ni, dim=ct.default_dim)])
        log.debug(self.collection_w.num_entities)

        # delete and query
        expr = f'{ct.default_int64_field_name} in [{insert_res.primary_keys[10]}]'
        self.collection_w.delete(expr=expr)
        self.collection_w.query(expr=expr, check_task=CheckTasks.check_query_empty, timeout=120)

        # compact
        self.collection_w.compact()
        self.collection_w.wait_for_compaction_completed()
        self.collection_w.get_compaction_plans()

        # drop index and re-create index
        self.collection_w.release()
        self.collection_w.drop_index(index_name=ct.default_index_name)
        self.collection_w.create_index(ct.default_float_vec_field_name, default_index_params,
                                       index_name=ct.default_index_name, timeout=3000)
        assert self.collection_w.indexes[0].params == default_index_params

        # search
        self.collection_w.load()
        search_res, _ = self.collection_w.search(query_vectors,
                                                 ct.default_float_vec_field_name,
                                                 ct.default_search_params, ct.default_limit)
        assert len(search_res) == ct.default_nq
        assert len(search_res[0]) == ct.default_limit

    def test_index_after_upgrade(self):
        """
        target: test do handoff search with old image, create index for data with new image
        method: before upgrade: init -> load -> index -> insert -> handoff -> search -> drop index
        expected: all succ
        """
        # init collection
        fields = [cf.gen_int64_field(is_primary=True), cf.gen_float_vec_field()]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)
        self.collection_w.init_collection(name=self.collection_name, schema=schema)

        # load collection
        self.collection_w.load()

        # create index
        # self.collection_w.create_index(ct.default_float_vec_field_name, default_index_params,
        #                                index_name=ct.default_index_name, timeout=360)
        # assert self.collection_w.indexes[0].params == default_index_params
        # log.debug(f"collection index: {self.collection_w.indexes[0].params}")

        # insert data
        for i in range(nb // ni):
            data = [cf.gen_vectors(nb=ni, dim=ct.default_dim)]
            self.collection_w.insert(data)
            log.debug(f"num entities: {self.collection_w.collection.num_entities}")

        # search
        search_res, _ = self.collection_w.search(cf.gen_vectors(nb=ct.default_nq, dim=ct.default_dim),
                                                 ct.default_float_vec_field_name,
                                                 ct.default_search_params, ct.default_limit)
        assert len(search_res) == ct.default_nq
        assert len(search_res[0]) == ct.default_limit

        # drop index
        # self.collection_w.drop_index()
        # assert self.collection_w.has_index()[0] is False

    def test_check_index_after_upgrade(self):
        """
        target: test do handoff search with old image, create index for data with new image
        method: after upgrade: create index -> delete -> query -> search
        expected: all succ
        """
        # describe index
        self.collection_w.init_collection(name=self.collection_name)
        log.debug(f"num entities: {self.collection_w.num_entities}")
        assert self.collection_w.has_index()[0] is False

        # create index
        new_index_params = {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 200}}
        self.collection_w.create_index(ct.default_float_vec_field_name, new_index_params,
                                       index_name=ct.default_index_name, timeout=3600)
        assert self.collection_w.indexes[0].params == new_index_params
        log.debug(f"collection index: {self.collection_w.indexes[0].params}")

        # load
        self.collection_w.load()

        # new insert, handoff
        data = [cf.gen_vectors(nb=nb, dim=ct.default_dim)]
        insert_res, _ = self.collection_w.insert(data)
        log.debug(f"num entities: {self.collection_w.num_entities}")

        # delete and query
        expr = f'{ct.default_int64_field_name} in {insert_res.primary_keys[0:10]}'
        self.collection_w.delete(expr=expr)
        self.collection_w.query(expr=expr, timeout=120, check_task=CheckTasks.check_query_empty)

        exp_query_res = [{ct.default_int64_field_name: insert_res.primary_keys[-1],ct.default_float_vec_field_name: data[0][-1]}]
        query_res, _ = self.collection_w.query(expr=f"{ct.default_int64_field_name} in [{insert_res.primary_keys[-1]}]",
                                               output_fields=[ct.default_float_vec_field_name],
                                               check_task=CheckTasks.check_query_results,
                                               check_items={"exp_res": exp_query_res, "with_vec": True})
        # search
        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        for i in range(20):
            search_res, _ = self.collection_w.search(cf.gen_vectors(nb=5, dim=ct.default_dim),
                                                     ct.default_float_vec_field_name,
                                                     search_params, ct.default_limit)
            assert len(search_res) == 5
            assert len(search_res[0]) == ct.default_limit
