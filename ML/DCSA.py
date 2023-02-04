import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

plt.rc('font', family='Times New Roman', size=7.5, weight='bold')
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 7.5}

class DCSAFeature:
    def __init__(self, train_data, val_data, test_data, train_label, val_label, test_label) -> None:
        self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
        self.train_label, self.val_label, self.test_label = train_label, val_label, test_label
        self.D0 = {'Ni':1.9e-4, 'Al': 1.85e-4, 'Co':7.5e-5, 'Cr':3e-6, 'Mo':1.15e-4, 'Re':8.2e-7, 
                   'Ru':2.48e-4, 'Ti':4.1e-4, 'Ta':0.031e-4, 'W':8e-6}
        self.Q_Ni = {'Ni': -287, 'Al': -284, 'Co': -284.169, 'Cr': -287, 'Mo': -267.585, 'Re': -278.817,
                     'Ru': -304.489, 'Ti': -256.9, 'Ta': -267.729, 'W': -282.13, 'Hf': -251.956}
        self.Gi = {'Al': -0.512, 'Co': -0.405, 'Cr': -0.488, 'Mo': -0.345, 'Re': -0.33,
                   'Ru': -0.21, 'Ti': -0.472, 'Ta': -0.472, 'W': -0.278, 'Hf': -0.716}
        self.Gi_Ni = {'Al': -0.41, 'Co': 0.845, 'Cr': -2.048, 'Mo': -2.16, 'Re': -2.4,
                      'Ru': -1.6, 'Ti': -0.8, 'Ta': -2.4, 'W': -2.16, 'Hf': -3.072}
        self.Gi_Al = {'Ni': 0.154, 'Cr': 0.512, 'Mo': 0.947, 'Re': 1.20,
                      'Ru': 0.24, 'Ti': 0.742, 'Ta': 0.512, 'W': 0.794, 'Hf': -0.768}
        self.features = ['温度/℃', '应力/Mpa', 'Ni', 'Al', 'Co', 'Cr', 'Mo', 'Re', 'Ru', 'Ti', 'Ta',
                         'W', 'Hf', 'Nb', 'Si', 'C', 'Y', 'Ce', 'B', 'gamapie', 'mismatch', 'APB', 'SFE', 'D', 'G']

    def D_calc(self, data):
        Qm = 0
        for a, n in self.Q_Ni.items():
            Qm += data[a + '_A1'] * n
        D_0 = 0
        for a, n in self.D0.items():
            D_0 += data[a + '_A1'] * n
        data['D'] =  D_0 * np.exp(Qm * 1000 / 8.314 / (data['温度/℃'] + 273.15))

    def G_clac(self, data):
        T = data['温度/℃'] + 273.15
        G_Ni_T = 103.2 - 3.102*10**(-2)*(T - 300) - 7.6262*10**(-6)*(T - 300)**2
        G_Ni3Al_T = 78 - 2.04*10**(-2)*(T - 300)
        G_var_Ni = 0
        for a, n in self.Gi.items():
            G_var_Ni += data[a + '_A1'] * n
        G_var_Ni3Al = 0
        for a, n in self.Gi_Ni.items():
            G_var_Ni3Al += 0.75 * data[a + '_L12'] * n
        for a, n in self.Gi_Al.items():
            G_var_Ni3Al += 0.25 * data[a + '_L12'] * n
        data['G'] = (1 - data['gamapie']) * G_Ni_T * (1 + G_var_Ni) + data['gamapie'] * G_Ni3Al_T * (1 + G_var_Ni3Al)
    
    def expend(self):
        # train
        self.D_calc(self.train_data)
        self.G_clac(self.train_data)
        self.train_data = self.train_data[self.features]
        # val
        self.D_calc(self.val_data)
        self.G_clac(self.val_data)
        self.val_data = self.val_data[self.features]
        # test
        self.D_calc(self.test_data)
        self.G_clac(self.test_data)
        self.test_data = self.test_data[self.features]


class DSCA(DCSAFeature):
    def __init__(self, train_data, val_data, test_data, train_label, val_label, test_label, n_cluster) -> None:
        super(DSCA, self).__init__(train_data, val_data, test_data, train_label, val_label, test_label)
        self.n_cluster = n_cluster
        self.train_k = None
        self.val_k = None
        self.test_k = None
        self.model_name = ['gpr', 'svr', 'rf', 'linear', 'lasso', 'ridge']
        self.model_list = [GaussianProcessRegressor(kernel=C(1, (0.01, 10)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.1, 2000)),
                           alpha=0.01, n_restarts_optimizer=10), 
                           SVR(kernel='rbf', C=100,  gamma='auto'), 
                           RandomForestRegressor(n_estimators=1000, max_depth=12, criterion='mae', bootstrap=True), 
                           LinearRegression(normalize=True), 
                           Lasso(alpha=0.01), 
                           Ridge(alpha = 0.02)]
        

    def min_max_transform(self):
        min_max = MinMaxScaler()
        self.train_data = min_max.fit_transform(self.train_data)
        self.val_data = min_max.transform(self.val_data)
        self.test_data = min_max.transform(self.test_data)
    
    def cluster_visual(self, data, k_label):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        x, y = pca_data[:, [0]], pca_data[:, [1]]
        c_list = ['#FF00FF', 'r', 'g', 'b', '#696969', '#FFA500', '#00BFFF', '#7CFC00', 'blueviolet', 'crimson', 'violet', 'gold', 'tan', 'yellow', 'brown']
        plt.figure(figsize=(5, 3.5))
        for index in range(self.n_cluster):
            c_index = np.argwhere(k_label==index).reshape(-1)
            plt.scatter(x[c_index, :], y[c_index, :], c=c_list[index], label='Cluster {}'.format(index + 1), s=8, alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 6})
        plt.show()
    
    def dsca_cluster(self):
        k_mean = KMeans(n_clusters=self.n_cluster, random_state=2021)
        k_mean.fit(self.train_data)
        self.train_k = k_mean.labels_
        self.cluster_visual(self.train_data, self.train_k)
        self.val_k, self.test_k = k_mean.predict(self.val_data), k_mean.predict(self.test_data)

    def dsca(self):
        cluster_score = []
        for index in range(self.n_cluster):
            c_index0 = np.argwhere(self.train_k==index).reshape(-1)
            c_index1 = np.argwhere(self.val_k==index).reshape(-1)
            c_train, c_train_label = self.train_data[c_index0, :], self.train_label.reshape(-1, 1)[c_index0, :]
            c_val, c_val_label = self.val_data[c_index1, :], self.val_label.reshape(-1, 1)[c_index1, :]
            val_score = []
            for model in self.model_list:
                model.fit(c_train, c_train_label)
                val_score.append(mean_absolute_error(c_val_label, model.predict(c_val)))
            cluster_score.append(val_score)
        return cluster_score

    def opt_dsca(self):
        cluster_score = self.dsca()
        best_cluster_model = []
        for i in range(self.n_cluster):
            best_cluster_model.append(cluster_score[i].index(min(cluster_score[i])))
            print('Cluster {} Best Model is {}'.format(i + 1, self.model_name[best_cluster_model[i]]))
        return best_cluster_model

    def pre_dsca(self, best_cluster_model, val_data, val_label, val_k):
        val_pre_list, val_label_list = [], []
        for index in range(self.n_cluster):
            c_index0 = np.argwhere(self.train_k==index).reshape(-1)
            c_index1 = np.argwhere(val_k==index).reshape(-1)
            c_train, c_train_label = self.train_data[c_index0, :], self.train_label.reshape(-1, 1)[c_index0, :]
            c_val, c_val_label = val_data[c_index1, :], val_label.reshape(-1, 1)[c_index1, :]
            val_score = []
            model = self.model_list[best_cluster_model[index]]
            model.fit(c_train, c_train_label)
            val_pre_list.append(np.array(model.predict(c_val)).reshape(-1, 1))
            val_score.append(mean_absolute_error(c_val_label, model.predict(c_val)))
            val_label_list.append(c_val_label)
        return np.concatenate(val_label_list, axis=0), np.concatenate(val_pre_list, axis=0)

    def main(self):
        self.min_max_transform()
        self.dsca_cluster()
        best_cluster_model = self.opt_dsca()
        train_life, train_pre = self.pre_dsca(best_cluster_model, self.train_data, self.train_label, self.train_k)
        val_life, val_pre = self.pre_dsca(best_cluster_model, self.val_data, self.val_label, self.val_k)
        test_life, test_pre = self.pre_dsca(best_cluster_model, self.test_data, self.test_label, self.test_k)
        return train_life, train_pre, val_life, val_pre, test_life, test_pre