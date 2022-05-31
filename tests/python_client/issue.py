from pymilvus import (
    connections,
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Index
)
import numpy as np

if __name__ == '__main__':

    # configure milvus hostname and port
    print(f"\nCreate connection...")
    connections.connect(host="10.98.0.9", port=19530)

    # List all collection names
    print(f"\nList collections...")
    print(list_collections())

    # Create a collection named 'demo_film_tutorial'
    print(f"\nCreate collection...")
    dim = 512
    field1 = FieldSchema(name="release_year", dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector", dim=dim, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name='demo_film_tutorial', data=None, schema=schema)

    print(f"\nInsert...")
    num = 50000
    # num = 50
    data = [
        [i for i in range(num)],
        np.random.random([num, dim]).tolist(),
    ]
    collection.insert(data)
    print(collection.num_entities)

    print(f"\nCreate index...")
    index_params = {"index_type": "ANNOY", "metric_type": "IP", "params": {"n_trees": 10}}
    index = Index(collection, "embedding", index_params)
    print(index.params)

    print([index.params for index in collection.indexes])