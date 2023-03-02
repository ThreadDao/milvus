import random
import multiprocessing
import time

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search


_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'vector_field'

# Vector parameters
_DIM = 512

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 128
_NPROBE = 16
_TOPK = 3


# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


# Create a collection named 'demo'
def create_collection(name, id_field, vector_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, is_primary=True, auto_id=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=_DIM)
    schema = CollectionSchema(fields=[field1, field2])
    collection = Collection(name=name, data=None, schema=schema, shards_num=2)
    print("\ncollection created:", name)
    return collection


def has_collection(name):
    return utility.has_collection(name)


# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())


def get_entity_num(collection):
    print("\nThe number of entity:")
    print(collection.num_entities)


def create_index(collection, filed_name):
    index_param = {
        "index_type": 'HNSW',
        "params": {"M": 64, 'efConstruction': 512},
        "metric_type": 'IP'}
    collection.create_index(filed_name, index_param)
    print("\nCreated index:\n{}".format(collection.index().params))


def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")


def load_collection(collection):
    collection.load()


def release_collection(collection):
    collection.release()
    print("\nLoad collection successfully")


def search(collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": 'IP', "params": {"ef": 32}},
        "limit": 10}
    results = collection.search(**search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))

def insert(start_part, end_part, repeat):
    print("process", start_part, "-", end_part, "begin to run")
    num = 5000
    data = [
        [[random.random() for _ in range(_DIM)] for _ in range(num)],
    ]

    collection = Collection(name=_COLLECTION_NAME)
    for n in range(repeat):
        for i in range (start_part, end_part):
            collection.insert(partition_name="part_"+str(i), data=data)
            # print("insert to", "part_"+str(i))

        print("process", start_part, "-", end_part, "inserted:", n)

    print("finish insert from", start_part, "to", end_part, "rows=",num*(end_part-start_part)*repeat)


if __name__ == '__main__':
    create_connection()

    part_num = 1000
    each_part_count = 100

    # collection = Collection(name=_COLLECTION_NAME)

    # num = 5000
    # data = [
    #     [[random.random() for _ in range(_DIM)] for _ in range(num)],
    # ]
    # for i in range(1):
    #     collection.insert(partition_name="part_" + str(i), data=data)
    #     print("insert to", "part_" + str(i))

    # collection.flush()
    # while True:
    #     print("total rows:", collection.num_entities)
    #     time.sleep(10)

    # insert(0, 1000, 1)
    # collection.drop()

    if has_collection(_COLLECTION_NAME):
        drop_collection(_COLLECTION_NAME)

    collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)
    print("collection created")

    index_param = {
        "index_type": 'FLAT',
        "params": {},
        "metric_type": 'L2'}
    collection.create_index(_VECTOR_FIELD_NAME, index_param)

    start = time.time()
    for i in range(part_num):
        collection.create_partition(partition_name="part_"+str(i))
        print("partition created", i)
    print("partitions created")
    end = time.time()
    print("partition cost:", end - start, "seconds")

    insert(0, 1000, 10)

    start = time.time()
    proc_num = part_num/float(each_part_count)
    print("process number", int(proc_num))
    process_list = []
    for i in range(int(proc_num)):
        p = multiprocessing.Process(target=insert, args=(i*each_part_count, (i+1)*each_part_count, 100))
        p.start()
        process_list.append(p)
        print("create process", i)

    print("wait processes...")
    for p in process_list:
        p.join()

    end = time.time()
    print("insert cost:", end - start, "seconds")

    start = time.time()
    collection.flush()
    end = time.time()
    print("total rows:", collection.num_entities, "flush cost:", end-start, "seconds")

