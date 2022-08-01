from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from base.client_base import TestcaseBase
from base.collection_wrapper import ApiCollectionWrapper
from base.schema_wrapper import ApiCollectionSchemaWrapper
from common import common_func as cf
from utils.util_log import test_log as log


class TestIssue(TestcaseBase):
    def test_issue_18480(self):
        default_search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        # connect
        self._connect()

        # collection fields and schema
        int_fields = cf.gen_int64_field(name="pk", is_primary=True)
        timestam_fields = cf.gen_double_field(name="timestam")
        vec_fields = cf.gen_float_vec_field(name="feature", dim=3840)
        fields = [int_fields, timestam_fields, vec_fields]

        def do_collection():
            schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)
            collection_w = ApiCollectionWrapper()
            collection_w.init_collection(name=cf.gen_unique_str("issue_"), schema=schema, shards_num=16)
            for i in range(10):
                df = pd.DataFrame({
                    "timestam": pd.Series(data=[np.double(i) for i in range(0, 1000)], dtype="double"),
                    "feature": cf.gen_vectors(1000, dim=3840)
                })
                insert_res, _ = collection_w.insert(df, timeout=360)
            log.debug(collection_w.num_entities)

            collection_w.load(timeout=360)
            collection_w.query(expr=f"pk in {insert_res.primary_keys[0:10]}", timeout=360)
            collection_w.search(cf.gen_vectors(2, dim=3840), "feature",
                                default_search_params, 10, timeout_decorator=360)

        tasks = []
        with ThreadPoolExecutor(max_workers=8) as t:
            for i in range(20):
                task = t.submit(do_collection)
                tasks.append(task)

        for task in tasks:
            task.done()

    def _test_tmp_collection(self):
        # connect
        self._connect()

        # collection fields and schema
        int_fields = cf.gen_int64_field(name="int", is_primary=True)
        double_fields = cf.gen_double_field(name="double")
        vec_fields = cf.gen_float_vec_field(name="vec", dim=256)
        fields = [int_fields, double_fields, vec_fields]
        schema, _ = ApiCollectionSchemaWrapper().init_collection_schema(fields=fields, auto_id=True)
