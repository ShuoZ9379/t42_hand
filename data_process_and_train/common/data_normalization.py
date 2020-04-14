import copy
import pdb

def min_max_normalize(data, min_arr, max_arr):
    normalized_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        normalized_data[i] = (data[i] - min_arr[:data.shape[1]]) / (max_arr[:data.shape[1]] - min_arr[:data.shape[1]])
    return normalized_data


def min_max_denormalize(data, min_arr, max_arr):
    denormalized_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        denormalized_data[i] = data[i] * (max_arr[:data.shape[1]] - min_arr[:data.shape[1]]) + min_arr[:data.shape[1]]
    return denormalized_data


def z_score_normalize(data, mean_arr, std_arr):
    normalized_data = data*0
    normalized_data = (data - mean_arr[:data.shape[-1]]) / std_arr[:data.shape[-1]]

    # normalized_data = copy.deepcopy(data)
    # for i in range(data.shape[0]):
    #     normalized_data[i] = (data[i] - mean_arr[:data.shape[-1]]) / std_arr[:data.shape[-1]]
    return normalized_data


def z_score_denormalize(data, mean_arr, std_arr):
    denormalized_data = data*0
    denormalized_data = data * std_arr[:data.shape[-1]] + mean_arr[:data.shape[-1]]
    # denormalized_data = copy.deepcopy(data)
    # for i in range(data.shape[0]):
    #     denormalized_data[i] = data[i] * std_arr[:data.shape[-1]] + mean_arr[:data.shape[-1]]
    return denormalized_data


def z_score_norm_single(data, mean_arr, std_arr):
    # pdb.set_trace()
    normalized_data = (data - mean_arr[:data.shape[-1]]) / std_arr[:data.shape[-1]]
    return normalized_data

    # normalized_data = data

def z_score_denorm_single(data, mean_arr, std_arr):
    denormalized_data = data * std_arr[:data.shape[-1]] + mean_arr[:data.shape[-1]]
    return denormalized_data
