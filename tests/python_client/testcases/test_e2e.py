import time
import pytest
from pymilvus import connections, utility, Collection, MilvusException, DataType

from base.client_base import TestcaseBase
from common import common_func as cf
from common import common_type as ct
from common.common_type import CaseLabel
from utils.util_log import test_log as log

prefix = "e2e_"

def get_loop_ids(start, end, batch, pk_type, varchar_len=64):
    if pk_type== DataType.INT64:
        while True:
            batch = min(batch, end - start)
            if start >= end:
                yield None
            ids = [i for i in range(start, start + batch)]
            start = start + len(ids)
            yield ids
    else:
        while True:
            batch = min(batch, end - start)
            if start >= end:
                yield None
            ids = [str(i).ljust(varchar_len, '0') for i in range(start, start + batch)]
            start = start + len(ids)
            yield ids

class TestE2e(TestcaseBase):
    """ Test case of end to end"""

    def setup_method(self, method):
        super().setup_method(method)

        # connect to server before testing
        self._connect()

    def find_collection_and_pk(self):
        #  find collection
        collections = utility.list_collections()
        if len(collections) == 0:
            raise Exception("No any collections, please create collection first!!!")
        log.info(f"collections: {collections}")
        collection_name = collections[0]

        # find pk type and name
        pk_name = "id"
        pk_type = DataType.INT64
        c = Collection(name=collection_name)
        for _field in c.schema.fields:
            if _field.is_primary:
                pk_name = _field.name
                pk_type = _field.dtype
        return collection_name, pk_name, pk_type

    @pytest.mark.tags(CaseLabel.L1)
    def test_milvus_default(self):
        # create
        name = cf.gen_unique_str(prefix)
        t0 = time.time()
        collection_w = self.init_collection_wrap(name=name, active_trace=True)
        tt = time.time() - t0
        assert collection_w.name == name

        # index
        index_params = {"index_type": "IVF_SQ8", "params": {"nlist": 64}, "metric_type": "L2"}
        t0 = time.time()
        index, _ = collection_w.create_index(field_name=ct.default_float_vec_field_name,
                                             index_params=index_params,
                                             index_name=cf.gen_unique_str())
        index, _ = collection_w.create_index(field_name=ct.default_string_field_name,
                                             index_params={},
                                             index_name=cf.gen_unique_str())
        tt = time.time() - t0
        log.info(f"assert index: {tt}")
        assert len(collection_w.indexes) == 2

        entities = collection_w.num_entities
        log.info(f"assert create collection: {tt}, init_entities: {entities}")

        # insert
        data = cf.gen_default_list_data()
        t0 = time.time()
        _, res = collection_w.insert(data)
        tt = time.time() - t0
        log.info(f"assert insert: {tt}")
        assert res

        # flush
        t0 = time.time()
        _, check_result = collection_w.flush(timeout=180)
        assert check_result
        assert collection_w.num_entities == len(data[0]) + entities
        tt = time.time() - t0
        entities = collection_w.num_entities
        log.info(f"assert flush: {tt}, entities: {entities}")

        # load
        collection_w.load()

        # search
        search_vectors = cf.gen_vectors(1, ct.default_dim)
        search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
        t0 = time.time()
        res_1, _ = collection_w.search(data=search_vectors,
                                       anns_field=ct.default_float_vec_field_name,
                                       param=search_params, limit=1)
        tt = time.time() - t0
        log.info(f"assert search: {tt}")
        assert len(res_1) == 1

        # release
        collection_w.release()

        # insert
        d = cf.gen_default_list_data()
        collection_w.insert(d)

        # search
        t0 = time.time()
        collection_w.load()
        tt = time.time() - t0
        log.info(f"assert load: {tt}")
        nq = 5
        topk = 5
        search_vectors = cf.gen_vectors(nq, ct.default_dim)
        t0 = time.time()
        res, _ = collection_w.search(data=search_vectors,
                                     anns_field=ct.default_float_vec_field_name,
                                     param=search_params, limit=topk)
        tt = time.time() - t0
        log.info(f"assert search: {tt}")
        assert len(res) == nq
        assert len(res[0]) <= topk
        # query
        term_expr = f'{ct.default_int64_field_name} in [1, 2, 3, 4]'
        t0 = time.time()
        res, _ = collection_w.query(term_expr)
        tt = time.time() - t0
        log.info(f"assert query result {len(res)}: {tt}")
        assert len(res) >= 4

    @pytest.mark.parametrize("size", [10000000])
    @pytest.mark.parametrize("start", [0])
    @pytest.mark.parametrize("batch", [60000])
    def test_delete_with_batch(self, size, start, batch):
        collection_name, pk_name, pk_type = self.find_collection_and_pk()
        # count before delete
        if pk_type == DataType.INT64:
            count_all_expr = f"{pk_name} >= 0"
        else:
            count_all_expr = f'{pk_name} != ""'

        # count before
        c = Collection(name=collection_name)
        count_all_before = c.query(count_all_expr, output_fields=["count(*)"], consistency_level="Strong")
        log.info(f"num entities before delete: {count_all_expr} with expr {count_all_before}")

        # delete size pks with rate
        for ids in get_loop_ids(start, start + size, batch, pk_type):
            if ids is None:
                break
            log.info(f"start to delete [{ids[0]}, ..., {ids[-1]}] with length {len(ids)}")
            start_time = time.time()
            delete_res = c.delete(expr=f"{pk_name} in {ids}")
            cost = time.time() - start_time
            log.info(f"delete cost {cost} with res {delete_res}")

    def test_delete_with_expr_retry(self):
        collection_name, pk_name, pk_type = self.find_collection_and_pk()
        count_all_expr = ""
        delete_expr = ""

        if pk_type == DataType.INT64:
            count_all_expr = f"{pk_name} >= 0"
            delete_expr = f"{pk_name} < 60000000"
        else:
            count_all_expr = f'{pk_name} != ""'
            delete_expr = f'{pk_name} < "60000000"'

        # count before
        c = Collection(name=collection_name)
        count_all_before = c.query(count_all_expr, output_fields=["count(*)"], consistency_level="Strong")
        log.info(f"total num before delete: {count_all_expr} with expr {count_all_before}")

        count_before = c.query(delete_expr, output_fields=["count(*)"], consistency_level="Strong")
        log.info(f"delete expr num before delete: {count_before} with expr {delete_expr}")

        log.info(f"start to delete with expr {delete_expr}")
        start = time.time()
        while True:
            try:
                loop_start = time.time()
                res = c.delete(delete_expr)
                loop_cost = time.time() - loop_start
                log.info(f"delete {res} loop_cost {loop_cost}s")
                break
            except MilvusException as me:
                log.warning(str(me))
        cost = time.time() - start
        log.info(f"total_cost {cost}s")

        #  count after
        count_all_after = c.query(count_all_expr, output_fields=["count(*)"], consistency_level="Strong")
        log.info(f"total num before delete: {count_all_expr} with expr {count_all_after}")

        count_after = c.query(delete_expr, output_fields=["count(*)"], consistency_level="Strong")
        log.info(f"delete expr num before delete: {count_after} with expr {delete_expr}")
