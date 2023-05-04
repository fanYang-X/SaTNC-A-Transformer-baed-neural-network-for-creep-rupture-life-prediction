import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# s electron number
s_dict = {'Ni': 2, 'Al': 2, 'Co': 2, 'Cr': 1, 'Mo': 1, 'Re': 2, 'Ru': 1, 'Ti': 2, 'Ta': 2, 'W': 2,
          'Hf': 2, 'Nb': 1, 'Si': 2, 'C': 2, 'Y': 2, 'Ce': 2, 'B': 2}
# d electron number
d_dict = {'Ni': 8, 'Al': 0, 'Co': 7, 'Cr': 5, 'Mo': 5, 'Re': 5, 'Ru': 7, 'Ti': 2, 'Ta': 3, 'W': 4,
          'Hf': 2, 'Nb': 4, 'Si': 0, 'C': 0, 'Y': 1, 'Ce': 1, 'B': 0}
# f electron number
f_dict = {'Ni': 0, 'Al': 0, 'Co': 0, 'Cr': 0, 'Mo': 0, 'Re': 14, 'Ru': 0, 'Ti': 0, 'Ta': 14, 'W': 14,
          'Hf': 14, 'Nb': 0, 'Si': 0, 'C': 0, 'Y': 0, 'Ce': 1, 'B': 0}
# Atomic radius
atom_radius = {'Ni': 0.149, 'Al': 0.118, 'Co': 0.152, 'Cr': 0.166, 'Mo': 0.19, 'Re': 0.188, 'Ru': 0.178, 'Ti': 0.176,
               'Ta': 0.2, 'W': 0.193, 'Hf': 0.208, 'Nb': 0.198, 'Si': 0.111,
               'C': 0.067, 'Y': 0.212, 'Ce': 0.185, 'B': 0.087}

# Diffusion activation energy in Ni
q_Ni = {'Ni': -287, 'Al': -284, 'Co': -284.169, 'Cr': -287, 'Mo': -267.585, 'Re': -278.817, 'Ru': -304.489,
        'Ti': -256.9, 'Ta': -267.729, 'W': -282.13, 'Hf': -251.956, 'Nb': 0, 'Si': 0, 'C': 0, 'Y': 0, 'Ce': 1, 'B': 0}

# Diffusion activation energy in Ni3Al
q_Ni3Al = {'Ni': -303, 'Al': -258, 'Co': -325, 'Cr': -366, 'Mo': -493, 'Re': -467.5, 'Ru': -318.7, 'Ti': -468, 'W': 0,
           'Ta': -425, 'Hf': 0, 'Nb': 0, 'Si': 0, 'C': 0, 'Y': 0, 'Ce': 1, 'B': 0}


def p_norm(feature):
    return (np.array([i for i in feature]) / max([abs(i) for i in feature])).reshape(1, -1)


def np_to_tensor(x):
    return torch.from_numpy(x).type(torch.float)


def mask(emb):
    mask_tensor = torch.ones((emb.shape[0], 18, 11))
    for i in range(emb.shape[0]):
        for j in range(18):
            if emb[i, j, 0] == 0:
                mask_tensor[i, j, :] = 0
    return mask_tensor * emb


class SaInput:
    def __init__(self, data, fold_num):
        self.data = data
        self.fold_num = fold_num
        self.atom = ['Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta', 'W', 'Hf',
                     'Nb', 'Si', 'C', 'Y', 'Ce', 'B']

        self.gama = ['Ni_A1', 'Al_A1', 'Co_A1', 'Cr_A1', 'Mo_A1', 'Re_A1', 'Ru_A1',
                     'Ti_A1', 'Ta_A1', 'W_A1', 'Hf_A1']

        self.gama_pie = ['Ni_L12', 'Al_L12', 'Co_L12', 'Cr_L12', 'Mo_L12', 'Re_L12',
                         'Ru_L12', 'Ti_L12', 'Ta_L12', 'W_L12', 'Hf_L12']

        self.domain = ['Temperature', 'Stress', 'gamapie', 'mismatch', 'APB', 'SFE']

        self.p_list = [s_dict, d_dict, f_dict, atom_radius, q_Ni, q_Ni3Al]


    def normal(self, data):
        for feature in self.atom + self.gama + self.gama_pie:
            min_max = MinMaxScaler(feature_range=(data['train'][0][feature].min(), 1))
            data['train'][0].loc[:, feature] = min_max.fit_transform(data['train'][0][feature].values.reshape(-1, 1))
            data['val'][0].loc[:, feature] = min_max.transform(data['val'][0][feature].values.reshape(-1, 1))
            data['test'][0].loc[:, feature] = min_max.transform(data['test'][0][feature].values.reshape(-1, 1))
        stand_scalar = StandardScaler()
        data['train'][0].loc[:, self.domain] = stand_scalar.fit_transform(data['train'][0][self.domain].values)
        data['val'][0].loc[:, self.domain] = stand_scalar.transform(data['val'][0][self.domain].values)
        data['test'][0].loc[:, self.domain] = stand_scalar.transform(data['test'][0][self.domain].values)
        return data


    def p_embedding(self, data):
        p = np.zeros((data.shape[0], 18, 11))
        p[:, : -1, 0] = data[self.atom].values
        p[:, : -1, 1] = np.concatenate([data[self.gama].values, np.zeros((data.shape[0], 6))], axis=1)
        p[:, : -1, 2] = np.concatenate([data[self.gama_pie].values, np.zeros((data.shape[0], 6))], axis=1)
        for index, p_dict in enumerate(self.p_list):
            p[:, : -1, index + 3] = np.repeat(p_norm(np.array([i for i in p_dict.values()])), data.shape[0], axis=0)
        p[:, :, [9, 10]] = np.repeat(data[self.domain].values[:, [0, 1]], 18, axis=0).reshape(-1, 18, 2)
        p[:, -1, : -2] = 1
        return np_to_tensor(p)


    def get_id(self, data):
        def to_id(x):
            id_list = []
            for i in range(17):
                if x[i] != 0:
                    id_list.append(i)
                else:
                    id_list.append(18)
            id_list.append(17)
            return id_list
        atom_id = list(map(to_id, data[self.atom].values.tolist()))
        return torch.tensor(atom_id).type(torch.long)


    def main(self):
        sa_data = {i: {} for i in range(self.fold_num)}
        for fold in range(self.fold_num):
            normal_data = self.normal(self.data[fold])
            for state in ['train', 'val', 'test']:
                state_data = normal_data[state][0]
                sa_data[fold][state] = [[mask(self.p_embedding(state_data)), self.get_id(state_data),
                                         np_to_tensor(state_data[self.domain].values)],
                                         np_to_tensor(np.log(normal_data[state][1] + 1))]
        return sa_data