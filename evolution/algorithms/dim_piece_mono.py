# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import numpy as np
from evolution.algorithms import GeneticAlgorithm, Individual


logger = logging.getLogger(__file__)


class DimPieceMonoGeneticAlgorithm(GeneticAlgorithm):

    def preprocess_init_factors(self, factors: np.ndarray) -> np.ndarray:
        return np.concatenate([[0, 0], factors])

    def extract_factors(self, factors: np.ndarray) -> np.ndarray:
        return factors[2:]

    def mutate(self, indv: Individual) -> Individual:
        list_step = self.list_step
        evo_list = np.arange(1.0, 1.0 + self.scale + list_step, list_step)

        new_factors = indv.factors
        while ((Individual(new_factors) in self.history) or (not np.all(np.diff(new_factors[2:]) >= 0))):
            alpha_dim = new_factors[0]
            beta_dim = new_factors[1]
            std_dev = 2.0

            if np.random.rand() < 0.5:
                alpha_dim = int(np.random.normal(alpha_dim, std_dev))
            else:
                beta_dim = int(np.random.normal(beta_dim, std_dev))

            while not (0 <= alpha_dim < beta_dim < self.head_size // 2):
                if np.random.rand() < 0.5:
                    alpha_dim = int(np.random.normal(alpha_dim, std_dev)) * 1.0
                else:
                    beta_dim = int(np.random.normal(beta_dim, std_dev)) * 1.0

            new_factors = new_factors.copy()
            new_factors[0] = alpha_dim
            new_factors[1] = beta_dim

            for dim in range(2, new_factors.shape[0] - 2):
                if np.random.rand() < 0.3:
                    if dim <= alpha_dim + 2: 
                        new_factors[dim] = 1.0
                    elif dim > beta_dim + 2:
                        new_factors[dim] = self.scale
                    else:
                        if dim == 2 + 0:
                            evo_list_curr = np.arange(1.0, new_factors[dim + 1], list_step)
                        elif dim == 2 + new_factors.shape[0]-1:
                            evo_list_curr = np.arange(new_factors[dim - 1], evo_list.max() + list_step, list_step)
                        else:
                            evo_list_curr = np.arange(new_factors[dim - 1], new_factors[dim+1] + list_step, list_step)
                        if evo_list_curr.shape[0] > 0:
                            layer_index = np.random.randint(0, evo_list_curr.shape[0])
                            new_factors = new_factors.copy()
                            new_factors[dim] = evo_list_curr[layer_index]

        indv = self.make_indv(new_factors)
        self.history.append(indv)
        return indv

    def crossover(self, indv_1: Individual, indv_2: Individual) -> Individual:
        par_factors_1 = indv_1.factors
        par_factors_2 = indv_2.factors
        if np.allclose(par_factors_1, par_factors_2):
            return None
        new_factors = par_factors_1.copy()
        for _ in range(self.max_crossover_try):
            for i in range(new_factors.shape[0]):
                if np.random.rand() < 0.3:
                    new_factors = new_factors.copy()
                    if np.random.rand() < 0.5:
                        if i < 2:
                            new_factors[0] = par_factors_2[0]
                            new_factors[1] = par_factors_2[1]
                        else:
                            new_factors[i] = par_factors_2[i]  
                    if (Individual(new_factors) in self.history) or \
                        (new_factors[0] > new_factors[1]) or \
                        (not np.all(np.diff(new_factors[2:]) >= 0)):
                        continue
                    indv = self.make_indv(new_factors)
                    self.history.append(indv)
                    return indv
        return None
