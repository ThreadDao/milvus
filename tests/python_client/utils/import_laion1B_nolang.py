import time
import minio
import os
import argparse
import logging
from typing import List, Optional
from datetime import datetime
import sys
from pymilvus.bulk_writer import (
    bulk_import, get_import_progress
)
from pymilvus import (
    FieldSchema, CollectionSchema, DataType, MilvusClient
)
import random


# Configure logging with UTC time and line numbers
class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        utc_dt = datetime.utcfromtimestamp(record.created)
        if datefmt:
            return utc_dt.strftime(datefmt)
        return utc_dt.strftime('%Y-%m-%d %H:%M:%S UTC')


def setup_logging():
    # Create formatter
    formatter = UTCFormatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Create file handler
    log_file = f"/tmp/milvus_import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger


# Setup logging
logger = setup_logging()
logger.info(f"Log file created at: /tmp/milvus_import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")

# Default values
DEFAULT_MILVUS_URL = "localhost:19530"
DEFAULT_MINIO_ENDPOINT = "10.104.18.35:9000"
DEFAULT_BUCKET_NAME = "zong-sn-wp-op-98-8151"
DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"
DEFAULT_PARQUET_FILES_NUM = 2
VECTOR_DIM = 768


class Config:
    def __init__(self):
        self.milvus_url = f"http://{DEFAULT_MILVUS_URL}"
        self.minio_endpoint = DEFAULT_MINIO_ENDPOINT
        self.bucket_name = DEFAULT_BUCKET_NAME
        self.access_key = DEFAULT_ACCESS_KEY
        self.secret_key = DEFAULT_SECRET_KEY
        self.parquet_files_num = DEFAULT_PARQUET_FILES_NUM
        self.action = 'upload-import'
        self.prepare_clean = False
        self.create_index = True

    @classmethod
    def from_args(cls, args):
        config = cls()
        config.milvus_url = f"http://{args.milvus_url}"
        config.minio_endpoint = args.minio_endpoint
        config.bucket_name = args.bucket_name
        config.parquet_files_num = args.parquet_files_num
        config.action = args.action
        config.prepare_clean = args.prepare_clean
        config.create_index = args.create_index
        return config

    def __str__(self):
        config_items = [
            f"Milvus URL: {self.milvus_url}",
            f"MinIO Endpoint: {self.minio_endpoint}",
            f"Bucket Name: {self.bucket_name}",
            f"Parquet Files Number: {self.parquet_files_num}",
            f"Action: {self.action}",
            f"Prepare Clean: {self.prepare_clean}",
            f"Create Index: {self.create_index}"
        ]
        return "\nConfiguration:\n" + "\n".join(f"  {item}" for item in config_items)


def gen_random_name(prefix: str) -> str:
    timestamp = int(time.time())
    randint = random.randint(1000, 9999)
    return f"{prefix}_{timestamp}_{randint}"


def gen_parquet_files(num_files: int) -> List[str]:
    return [f"binary_768d_{i:05d}.parquet" for i in range(num_files)]


class MinioClient:
    def __init__(self, config: Config):
        self.config = config
        self.client = minio.Minio(
            endpoint=config.minio_endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=False
        )

    def upload_parquet_files(self) -> None:
        parquet_files = gen_parquet_files(self.config.parquet_files_num)

        for object_name in parquet_files:
            file_path = f"/test/milvus/raw_data/laion5b_parquet/laion1B_nolang/{object_name}"
            try:
                logger.info(f"Uploading {file_path}")
                self.client.fput_object(self.config.bucket_name, object_name, file_path)
                logger.info(f"Successfully uploaded {file_path} to s3://{self.config.bucket_name}/{object_name}")
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {str(e)}")
                raise


class MilvusHandler:
    def __init__(self, config: Config):
        self.config = config
        self.client = MilvusClient(uri=config.milvus_url)

    def get_collection_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="pk_5b", dtype=DataType.INT64, is_clustering_key=True),
            FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=8192, enable_analyzer=True,
                        enable_match=True),
            FieldSchema(name="NSFW", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="similarity", dtype=DataType.DOUBLE),
            FieldSchema(name="width", dtype=DataType.INT64, is_partition_key=True),
            FieldSchema(name="height", dtype=DataType.INT64),
            FieldSchema(name="original_width", dtype=DataType.INT64),
            FieldSchema(name="original_height", dtype=DataType.INT64),
            FieldSchema(name="md5", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="float32_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        ]
        return CollectionSchema(fields=fields, description="Collection from Parquet schema")

    def prepare_index(self, name):
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="float32_vector",  # Name of the vector field to be indexed
            index_type="HNSW",  # Type of the index to create
            index_name="vector_index",  # Name of the index to create
            metric_type="COSINE",  # Metric type used to measure similarity
            params={
                "M": 8,  # Maximum number of neighbors each node can connect to in the graph
                "efConstruction": 200
                # Number of candidate neighbors considered for connection during index construction
            }  # Index building params
        )
        index_params.add_index(
            field_name="NSFW",
            index_type="",
            index_name="varchar_index"
        )
        index_params.add_index(
            field_name="similarity",
            index_type="",
            index_name="float_index"
        )
        index_params.add_index(
            field_name="original_height",
            index_type="",
            index_name="int_index"
        )
        res = self.client.create_index(name, index_params)
        logger.info(f"create index res: {res}")

    def prepare_collection(self, prepare_clean: bool = False, _create_index: bool = True) -> str:
        if prepare_clean:
            collections = self.client.list_collections()
            logger.info(f"Dropping existing collections: {collections}")
            for collection in collections:
                self.client.drop_collection(collection)

        schema = self.get_collection_schema()
        name = gen_random_name("import")
        self.client.create_collection(name, schema=schema)
        logger.info(f"Created collection: {name}")

        if _create_index:
            self.prepare_index(name)
        return name

    def import_data(self, collection_name: str, parquet_files: List[str]) -> None:
        files = [[file] for file in parquet_files]
        try:
            resp = bulk_import(
                url=self.config.milvus_url,
                collection_name=collection_name,
                files=files
            )
            job_id = resp.json()['data']['jobId']
            logger.info(f"Started import job {job_id} for {parquet_files}")

            while True:
                progress_resp = get_import_progress(url=self.config.milvus_url, job_id=job_id)
                progress_data = progress_resp.json()["data"]
                state = progress_data['state']
                logger.info(f"Task {job_id} state: {state}, progress: {progress_data['progress']}, "
                            f"imported {progress_data['importedRows']} rows of total {progress_data['totalRows']}")

                if state == "Failed":
                    raise Exception(f"Import job {job_id} failed")
                if state == "Completed":
                    break
                time.sleep(3)
        except Exception as e:
            logger.error(f"Failed to import data: {str(e)}")
            raise


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description='Milvus data import tool')
    parser.add_argument('--milvus-url', type=str, default=DEFAULT_MILVUS_URL,
                        help=f'Milvus server URL (default: {DEFAULT_MILVUS_URL})')
    parser.add_argument('--minio-endpoint', type=str, default=DEFAULT_MINIO_ENDPOINT,
                        help=f'MinIO endpoint (default: {DEFAULT_MINIO_ENDPOINT})')
    parser.add_argument('--bucket-name', type=str, default=DEFAULT_BUCKET_NAME,
                        help=f'MinIO bucket name (default: {DEFAULT_BUCKET_NAME})')
    parser.add_argument('--parquet-files-num', type=int, default=DEFAULT_PARQUET_FILES_NUM,
                        help=f'Number of parquet files to process (default: {DEFAULT_PARQUET_FILES_NUM})')
    parser.add_argument('--action', type=str, choices=['only-upload', 'only-import', 'upload-import'],
                        default='upload-import', help='Action to perform')
    parser.add_argument('--prepare-clean', action='store_true',
                        help='Drop existing collections before import')
    parser.add_argument('--create-index', action='store_true',
                        help='Create index for imported collection')

    return Config.from_args(parser.parse_args())


def main():
    try:
        config = parse_args()
        logger.info(config)
        minio_client = MinioClient(config)
        milvus_handler = MilvusHandler(config)

        if config.action in ['only-upload', 'upload-import']:
            minio_client.upload_parquet_files()

        if config.action in ['only-import', 'upload-import']:
            collection_name = milvus_handler.prepare_collection(config.prepare_clean, config.create_index)
            parquet_files = gen_parquet_files(config.parquet_files_num)
            logger.info(f"Total import parquet files: {parquet_files}")
            milvus_handler.import_data(collection_name, parquet_files)
            logger.info("All tasks completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
