import json
import random
import pandas as pd
from sklearn import preprocessing

varchar_field = "varchar"
vector_field = "vec"


def gen_vectors(nb, dim):
    vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()


def gen_default_df(nb, dim, start=0):
    # int_values = pd.Series(data=[i for i in range(start, start + nb)])
    # float_values = pd.Series(data=[np.float32(i) for i in range(start, start + nb)], dtype="float32")
    string_values = pd.Series(data=[str(i) for i in range(start, start + nb)], dtype="string")
    float_vec_values = gen_vectors(nb, dim)
    df = pd.DataFrame({
        # ct.default_int64_field_name: int_values,
        # ct.default_float_field_name: float_values,
        varchar_field: string_values,
        vector_field: float_vec_values
    })
    return df

def gen_rows_data(nb, dim, start=0):
    data = {
        "rows": [
        ]
    }
    for i in range(start, start+nb):
        entity = {
            varchar_field: str(i),
            vector_field: [random.random() for _ in range(dim)] }
        data["rows"].append(entity)

    return data


if __name__ == '__main__':
    # # df = gen_default_df(nb=1, dim=2048)
    # data = gen_rows_data(10000, dim=2048, start=1000)
    # with open('/Users/nausicca/Documents/tmp_vdc/data_s.json', 'w') as f:
    #     json.dump(data, f)
    a = "4"
    b = "173"
    print(b > a)

