# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import numpy as np
from evolution.algorithms import GeneticAlgorithm, Individual


logger = logging.getLogger(__file__)


class DimMonoGeneticAlgorithm(GeneticAlgorithm):

    def mutate(self, indv: Individual) -> Individual:
        list_step = self.list_step
        def ceil_dim_value(dim_value, dim):
            step = self.list_step[dim]
            if self.dim_search_space is not None:
                dim_value = max(self.dim_search_space[dim][0], min(dim_value, self.dim_search_space[dim][1]))
            return dim_value // step * step + (0 if dim_value % step < 1e-10 else step)

        def floor_dim_value(dim_value, dim):
            step = self.list_step[dim]
            if self.dim_search_space is not None:
                dim_value = max(self.dim_search_space[dim][0], min(dim_value, self.dim_search_space[dim][1]))
            return dim_value // step * step

        new_factors = indv.factors
        while ((Individual(new_factors) in self.history) or (not np.all(np.diff(new_factors) > 0))):
            for dim in range(self.critical_dim, new_factors.shape[0]):
                if np.random.rand() < 0.3:
                    if dim == 0:
                        evo_list_curr = np.arange(ceil_dim_value(1.0, dim), floor_dim_value(new_factors[dim + 1], dim) + list_step[dim], list_step[dim])
                    elif dim == new_factors.shape[0] - 1:
                        evo_list_curr = np.arange(ceil_dim_value(new_factors[dim - 1], dim), floor_dim_value(new_factors[dim - 1] + 4, dim) + list_step[dim], list_step[dim])
                    else:
                        evo_list_curr = np.arange(ceil_dim_value(new_factors[dim - 1], dim), floor_dim_value(new_factors[dim + 1], dim) + list_step[dim], list_step[dim])

                    if evo_list_curr.shape[0] > 0:
                        layer_index = np.random.randint(0, evo_list_curr.shape[0])
                        new_factors = new_factors.copy()
                        new_factors[dim] = evo_list_curr[layer_index]
                    
                    def ntk_left(total_dim, cd_dim, scale, new_factors):
                        ext = scale ** (total_dim / cd_dim)
                        for i in range(cd_dim):
                            new_factors[i] = ext ** (i / total_dim)
                        return new_factors

                    new_factors = ntk_left(new_factors.shape[0], self.critical_dim, new_factors[self.critical_dim], new_factors)

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
            for i in range(self.critical_dim, new_factors.shape[0]):
                if np.random.rand() < 0.3:
                    new_factors = new_factors.copy()
                    if np.random.rand() < 0.5:
                        new_factors[i] = par_factors_2[i]
                    if (Individual(new_factors) in self.history) or (not np.all(np.diff(new_factors) > 0)):
                        continue

                    def ntk_left(total_dim, cd_dim, scale, new_factors):
                        ext = scale ** (total_dim / cd_dim)
                        for i in range(cd_dim):
                            new_factors[i] = ext ** (i / total_dim)
                        return new_factors

                    new_factors = ntk_left(new_factors.shape[0], self.critical_dim, new_factors[self.critical_dim], new_factors)

                    indv = self.make_indv(new_factors)
                    self.history.append(indv)
                    return indv
        return None
