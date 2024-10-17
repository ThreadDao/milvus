import time
from pathlib import Path

import pytest
from pymilvus import FieldSchema, DataType
from common.common_func import gen_unique_str

from base.client_base import TestcaseBase
from common.milvus_sys import MilvusSys
from utils.util_k8s import get_milvus_instance_name, get_milvus_deploy_tool, get_pod_ip_name_pairs
from utils.util_log import test_log as log

from common.minio_comm import copy_files_to_minio


class TestImport(TestcaseBase):

    @pytest.fixture(scope="function", autouse=True)
    def init_minio_client(self, host, milvus_ns="qa-milvus"):
        Path("/tmp/bulk_insert_data").mkdir(parents=True, exist_ok=True)
        self._connect()
        self.milvus_ns = milvus_ns
        self.milvus_sys = MilvusSys(alias='default')
        self.instance_name = get_milvus_instance_name(self.milvus_ns, host)
        self.deploy_tool = get_milvus_deploy_tool(self.milvus_ns, self.milvus_sys)
        minio_label = f"release={self.instance_name}, app=minio"
        if self.deploy_tool == "milvus-operator":
            minio_label = f"release={self.instance_name}-minio, app=minio"
        minio_ip_pod_pair = get_pod_ip_name_pairs(
            self.milvus_ns, minio_label
        )
        ms = MilvusSys()
        minio_ip = list(minio_ip_pod_pair.keys())[0]
        minio_port = "9000"
        self.minio_endpoint = f"{minio_ip}:{minio_port}"
        self.bucket_name = ms.index_nodes[0]["infos"]["system_configurations"][
            "minio_bucket_name"
        ]

    def init_collections(self):
        c_name = gen_unique_str("import_laion")

        self._connect()
        # fields
        f_pk = FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True)
        f_pk_5b = FieldSchema(name="pk_5b", dtype=DataType.INT64)
        f_caption = FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=8192)
        f_NSFW = FieldSchema(name="NSFW", dtype=DataType.VARCHAR, max_length=8192)
        f_similarity = FieldSchema(name="similarity", dtype=DataType.FLOAT)
        f_width = FieldSchema(name="width", dtype=DataType.FLOAT)
        f_height = FieldSchema(name="height", dtype=DataType.FLOAT)
        f_original_width = FieldSchema(name="original_width", dtype=DataType.FLOAT)
        f_original_height = FieldSchema(name="original_height", dtype=DataType.FLOAT)
        f_md5 = FieldSchema(name="md5", dtype=DataType.VARCHAR, max_length=8192)
        f_float32_vector = FieldSchema(name="float32_vector", dtype=DataType.FLOAT_VECTOR, dim=768)

        schema = self.collection_schema_wrap.init_collection_schema(
            fields=[f_pk, f_pk_5b, f_caption, f_NSFW, f_similarity, f_width, f_height, f_original_width,
                    f_original_height,
                    f_md5, f_float32_vector])[0]
        self.collection_wrap.init_collection(c_name, schema=schema)

        # create index
        default_index = {"index_type": "IVF_SQ8", "metric_type": "COSINE",
                         "params": {"nlist": 128}}
        self.collection_wrap.create_index(
            field_name="float32_vector", index_params=default_index
        )
        return c_name

    def test_import_laion(self):
        c_name = self.init_collections()

        # import and wait
        costs = []
        files = ["binary_768d_00000.parquet", "binary_768d_00002.parquet", "binary_768d_00001.parquet"]
        copy_files_to_minio(self.minio_endpoint, r_source="/test/milvus/raw_data/laion5b_parquet/laion1B_nolang/",
                        files=files, bucket_name=self.instance_name)
        for _file in files:
            t0 = time.time()
            task_id, _ = self.utility_wrap.do_bulk_insert(
                collection_name=c_name,
                partition_name=None,
                files=files,
            )
            log.info(f"bulk insert task id:{task_id}")
            success, _ = self.utility_wrap.wait_for_bulk_insert_tasks_completed(
                task_ids=[task_id], timeout=1200
            )
            tt = time.time() - t0
            costs.append(tt)
            log.info(f"bulk insert state:{success} in {tt}")
            assert success

        num_entities = self.collection_wrap.num_entities
        log.info(f" collection entities: {num_entities}")
        log.info(f"costs is {costs}, avg cost is {sum(costs)/3}")