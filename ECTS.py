### https://github.com/Eukla/ETS/tree/master/ets/algorithms

import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Sequence, Dict, Optional
import multiprocessing as mp

class ECTS():
    """Algorytm ECTS"""

    def __init__(self, timestamps, support: float):
        """
        Tworzy instancję ECTS.
        :param timestamps: lista znaczników czasu dla wczesnych prognoz
        :param support: minimalny próg wsparcia
        """
        self.rnn: Dict[int, Dict[int, List]] = dict()
        self.nn: Dict[int, Dict[int, List]] = dict()
        self.data: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.mpl: Dict[int, Optional[int]] = dict()
        self.timestamps = timestamps
        self.support = support
        self.clusters: Dict[int, List[int]] = dict()
        self.occur: Dict[int, int] = dict()
        self.correct: Optional[List[Optional[int]]] = None

    def train(self, train_data: pd.DataFrame, labels: Sequence[int]) -> None:
        """
        Trenowanie modelu.
        :param train_data: zbiór treningowy jako DataFrame
        :param labels: zbiór przypisanych klas do szeregów ze zbioru treningowego
        """
        self.data = train_data
        self.labels = labels

        for index, value in self.labels.value_counts().items():
            self.occur[index] = value

        time_pos = 0
        for e in self.timestamps:
            product = self.__nn_non_cluster(time_pos) 
            self.rnn[e] = product[1]
            self.nn[e] = product[0]
            time_pos += 1
        temp = {}
        finished = {}  
        for e in reversed(self.timestamps):
            for index, row in self.data.iterrows():
                if index not in temp:
                    self.mpl[index] = e
                    finished[index] = 0  

                else:
                    if finished[index] == 1: 
                        continue

                    if self.rnn[e][index] is not None:
                        self.rnn[e][index].sort()
                    if temp[index] is not None:
                        temp[index].sort()

                    if self.rnn[e][index] == temp[index]: 
                        self.mpl[index] = e

                    else:  
                        finished[index] = 1
                temp[index] = self.rnn[e][index]
        self.__mpl_clustering()

    def __nn_non_cluster(self, prefix: int):
        """Funkcja znajduje zbiór NN i RNN dla wszystkich szeregów czasowych o zadanej długości prefiksu.
        :param prefix: długość prefiksu
        :return: słowniki przechowujące zbiory NN i RNN"""
        nn = {}
        rnn = {}
        neigh = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(self.data.iloc[:, 0:prefix + 1])
        def something(row):
            return neigh.kneighbors([row])

        result_data = self.data.iloc[:, 0:prefix + 1].apply(something, axis=1)
        for index, value in result_data.items():
            if index not in nn:
                nn[index] = []
            if index not in rnn:
                rnn[index] = []
            for item in value[1][0]:
                if item != index:
                    nn[index].append(item)
                    if item not in rnn:
                        rnn[item] = [index]
                    else:
                        rnn[item].append(index)
        return nn, rnn

    def __cluster_distance(self, cluster_a: Sequence[int], cluster_b: Sequence[int]):
        """
        Funkcja oblicz odległość między dwoma klastami i szuka minimalnej odległości między wszystkimi parami elementów z dwóch klastrów.
        :param cluster_a: pierwszy klaster
        :param cluster_b: drugi klaster
        :return:  odległość
        """

        min_distance = float("inf")
        for i in cluster_a:
            for j in cluster_b:
                d = distance.euclidean(self.data.iloc[i], self.data.iloc[j])
                if min_distance > d:
                    min_distance = d

        return min_distance

    def nn_cluster(self, cl_key: int, cluster_index: Sequence[int]):
        """
        Funkcja szuka najbliższego klastra używając __cluster_distance.
        :param cluster_index: lista indeksów serii należących do tego klastra
        :param cl_key: klucz klastra w słowniku
        """
        dist = float("inf")
        candidate = [] 

        for key, value in self.clusters.items(): 

            if cl_key == key: 
                continue
            temp = self.__cluster_distance(cluster_index, value) 

            if dist >= temp: 
                dist = temp
                candidate = [key]
        return candidate

    def __rnn_cluster(self, e: int, cluster: List[int]):
        """
        Oblicza RNN klastra dla obecnegp prefiksu.
        :param e: prefiks, dla którego szukamy zbioru RNN
        :param cluster: klaster, dla którego szukamy zbioru RNN
        """

        rnn = set()
        complete = set()
        for item in cluster:
            rnn.union(self.rnn[e][item])
        for item in rnn:
            if item not in cluster:
                complete.add(item)
        return complete

    def __mpl_calculation(self, cluster: List[int]):
        """
        Funkcja szuka MPL dla klastrów.
        :param cluster: klaster, dla którego szukamy MPL
        """
        index = self.labels[cluster[0]]
        if self.support > len(cluster) / self.occur[index]:
            return #nie liczymy, jeśli klaster jest zbyt mały
        mpl_rnn = self.timestamps[len(self.timestamps) - 1] 
        mpl_nn = self.timestamps[len(self.timestamps) - 1]

        curr_rnn = self.__rnn_cluster(self.timestamps[len(self.timestamps) - 1], cluster)  # RNN dla pełniej długości

        for e in reversed(self.timestamps):
            temp = self.__rnn_cluster(e, cluster)  # RNN dla kolejnych długości
            if not curr_rnn - temp: 
                mpl_rnn = e
            else:
                break
            curr_rnn = temp

        rule_broken = 0
        for e in reversed(self.timestamps):  # NN dla kolejnych długości
            for series in cluster:  # Dla wszystkich szeregów czasowych
                for my_tuple in self.nn[e][series]:  
                    if my_tuple not in cluster:
                        rule_broken = 1
                        break
                if rule_broken == 1:
                    break
            if rule_broken == 1:
                break
            else:
                mpl_nn = e
        for series in cluster:
            pos = max(mpl_rnn, mpl_nn)  
            if self.mpl[series] > pos:
                self.mpl[series] = pos

    def __mpl_clustering(self):
        """Funkcja wywołuje grupowanie hierarchiczne"""
        n = self.data.shape[0]
        redirect = {}
        discriminative = 0

        # Każdy szereg jako osobny klaster
        for index, row in self.data.iterrows():
            self.clusters[index] = [index]
            redirect[index] = index

        result = []
        max_iterations = n * n  # maksymalna liczba iteracji, zabezpieczenie
        iter_count = 0

        while n > 1:
            iter_count += 1
            if iter_count > max_iterations:
                break

            closest = {}

            # Wyznaczamy najbliższy klaster dla każdego klastra
            for key, cluster in self.clusters.items():
                closest[key] = self.nn_cluster(key, cluster)

            merged = False

            for key, candidates in closest.items():
                for item in list(candidates):
                    if key in closest.get(item, []):
                        # Sprawdzenie, czy nie są już w tym samym klastrze
                        if redirect[item] == redirect[key]:
                            continue

                        self.clusters[redirect[key]] += self.clusters[redirect[item]]
                        del self.clusters[redirect[item]]
                        n -= 1
                        redirect[item] = redirect[key]

                        result = [self.labels.loc[idx] for idx in self.clusters[redirect[key]]]
                        if len(set(result)) == 1:
                            discriminative += 1
                            self.__mpl_calculation(self.clusters[redirect[key]])

                        for k in redirect:
                            if redirect[k] == item:
                                redirect[k] = redirect[key]

                        merged = True

            # Jeśli nie połączono żadnego klastra, kończymy pętlę
            if not merged:
                break

            discriminative = 0

    def predict(self, test_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Faza predykcji."""
        predictions = []
        nn = []
        candidates = [] 
        cand_min_mpl = []
        for test_index, test_row in test_data.iterrows():
            for e in self.timestamps:
                neigh = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.data.iloc[:, 0:e + 1])
                neighbors = neigh.kneighbors([test_row[0:e + 1]])
                candidates.clear()
                cand_min_mpl.clear()
                nn = neighbors[1]
                for i in nn:
                    if e >= self.mpl[i[0]]:
                        candidates.append((self.mpl[i[0]], self.labels[i[0]])) 
                if len(candidates) > 1: 
                    candidates.sort(key=lambda x: x[0])
                    for candidate in candidates:

                        if candidate[0] == candidates[0][0]:
                            cand_min_mpl.append(candidate) 
                        else:
                            break 
                    predictions.append((e, max(set(cand_min_mpl), key=cand_min_mpl.count)))  
                    break
                elif len(candidates) == 1: 
                    predictions.append((e, candidates[0][1]))
                    break
            if candidates == 0:
                predictions.append((self.timestamps[-1], 0))
        return predictions