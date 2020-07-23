
import pandas
import torch
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split

class normalization:

 def normal_data(file_name, train_rows_num):
    df = pandas.read_csv(file_name, delimiter=',')

    df.drop(df.columns[[0]], axis=1, inplace=True)
    # User list comprehension to create a list of lists from Dataframe rows
    list_of_rows = [list(row) for row in df.values]

    # Insert Column names as first list in list of lists
    list_of_rows.insert(0, df.columns.to_list())

    list_of_rows = np.array(list_of_rows, dtype=np.float32)

    # Print list of lists i.e. rows
    ##print(list_of_rows)
    new_data = []

    # delete 10
    train_rows_num = 10
    for k in range(train_rows_num):
        vector_16 = []
        list_a = []
        list_b = []
        list_c = []
        list_d = []
        for i in list_of_rows[k][0:: 4]:
            list_a.append(i)

        for i in list_of_rows[k][1:: 4]:
            list_b.append(i)

        for i in list_of_rows[k][2:: 4]:
            list_c.append(i)

        for i in list_of_rows[k][3:: 4]:
            list_d.append(i)

        vector_16.append(np.mean(list_a))
        vector_16.append(np.mean(list_b))
        vector_16.append(np.mean(list_c))
        vector_16.append(np.mean(list_d))

        vector_16.append(np.std(list_a))
        vector_16.append(np.std(list_b))
        vector_16.append(np.std(list_c))
        vector_16.append(np.std(list_d))

        vector_16.append(np.mean(np.abs(np.diff(list_a))))
        vector_16.append(np.mean(np.abs(np.diff(list_b))))
        vector_16.append(np.mean(np.abs(np.diff(list_c))))
        vector_16.append(np.mean(np.abs(np.diff(list_d))))

        vector_16.append(np.std(np.abs(np.diff(list_a))))
        vector_16.append(np.std(np.abs(np.diff(list_b))))
        vector_16.append(np.std(np.abs(np.diff(list_c))))
        vector_16.append(np.std(np.abs(np.diff(list_d))))

        new_data.append(vector_16)

    random.shuffle(new_data)

    mutate_dataset, training_dataset = sklearn.model_selection.train_test_split(new_data, train_size=0.5, test_size=0.5)

    mutate_1, mutate_2 = sklearn.model_selection.train_test_split(mutate_dataset, train_size=0.5, test_size=0.5)

    mutate_data_1, mutate_data_2 = sklearn.model_selection.train_test_split(mutate_1, train_size=0.5, test_size=0.5)
    mutate_data_3, mutate_data_4 = sklearn.model_selection.train_test_split(mutate_2, train_size=0.5, test_size=0.5)

    print(mutate_data_1)


    return torch.tensor(mutate_data_1)

