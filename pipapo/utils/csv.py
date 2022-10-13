import imp
import pandas
from .io import check_if_file_exist


def export_csv(dictionary, file_path):
    check_if_file_exist(file_path)
    labels = []
    data_arrays = []
    for label, data in dictionary.items():
        if data.shape[1] == 1:
            data_arrays.append(data.flatten())
            labels.append(label)
        else:
            for i, column in enumerate(data.T):
                data_arrays.append(column.flatten())
                labels.append(label + f"_{i}")

    pd_dataframe = pandas.DataFrame.from_dict(dict(zip(labels, data_arrays)))
    pd_dataframe.to_csv(file_path, sep=",", index=False)
