import os
import abc
import json
import logging
import mlflow

import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from evaluation.perplexity import compute_perplexity


logger = logging.getLogger(__file__)
mlflow.autolog()


class Individual(object):
    
    def __init__(self, factors: np.ndarray, ppl: float = None):
        self.factors = factors
        self.ppl = ppl

    def __eq__(self, other):
        return np.allclose(self.factors, other.factors)

    def __str__(self):
        return f'{self.factors.tolist()} => {self.ppl}'

    def evaluate(self, model: AutoModelForCausalLM, eval_args: dict):
        self.ppl = compute_perplexity(model=model, **eval_args)


class GeneticAlgorithm:

    def __init__(
        self,
        model: AutoModelForCausalLM,
        scale: float,
        target_length: int,
        hyper_params: dict[str, float],
        init_factors: np.ndarray,
        log_json_path: str,
        output_dir: str,
        eval_args: dict,
        recovery: str = None
    ):
        self.model = model
        self.scale = scale
        self.target_length = target_length

        self.history: list[Individual] = []

        evo_scale = hyper_params["evo_scale"]
        self.population_size = int(evo_scale * hyper_params["population_size"])      # 种群大小
        self.max_time_budget = int(evo_scale * hyper_params["max_time_budget"])      # 迭代伦次
        self.mutation_numbers = int(evo_scale * hyper_params["mutation_numbers"])    # 变异操作数量
        self.crossover_size = int(evo_scale * hyper_params["crossover_size"])        # 交叉操作数量
        self.max_crossover_try = int(evo_scale * hyper_params["max_crossover_try"])  # 交叉重试次数
        self.parents_size = int(evo_scale * hyper_params["parents_size"])            # 亲代数量
        self.list_step = hyper_params["list_step"]                                   # 搜索空间粒度
        assert self.parents_size <= self.population_size, \
            f'Number of parents ({self.parents_size}) should not be larger than population size ({self.population_size})'

        self.head_size = init_factors.shape[0] * 2
        self.init_factors = self.preprocess_init_factors(init_factors)

        self.eval_args = eval_args

        self.recovery = recovery
        self.log_json_path = log_json_path
        self.output_dir = output_dir

    def preprocess_init_factors(self, factors: np.ndarray) -> np.ndarray:
        return factors

    @abc.abstractmethod
    def mutate(self, indv: Individual) -> Individual:
        "Generate new individual with constraints by mutatation."

    @abc.abstractmethod        
    def crossover(self, indv_1: Individual, indv_2: Individual) -> Individual:
        "Generate new individual with constraints by crossover."

    @abc.abstractmethod
    def make_indv(self, factors: np.ndarray) -> Individual:
        "Evaluate generated factors and returns an individual."

    def log(self, iteration: int, population: list[Individual]):
        with open(self.log_json_path, 'w') as file:
            file.write(json.dumps(
                {
                    'iteration': iteration,
                    'population': [[indv.factors.tolist(), indv.ppl] for indv in population],
                    'history': [[indv.factors.tolist(), indv.ppl] for indv in self.history],
                },
                indent = 4,
            ))
        try: mlflow.log_metric("ppl", population[0].ppl, step=iteration)
        except: pass
        np.savetxt(
            os.path.join(self.output_dir, f"result_it{iteration:0>3d}.csv"), population[0].factors,
            delimiter='\n',
        )

    def run_genetic_algorithm(self):
        "Main loop of Genetic Algorithm."

        if self.recovery is None:
            population = []
            latest_iteration = 0
            indv = self.make_indv(self.init_factors)
            logger.info(f"[Population #{0:3d}]:{indv}")
            population.append(indv)
            self.history.append(indv)
            for i in range(self.population_size - 1):
                new_indv = self.mutate(indv)
                population.append(new_indv)
                logger.info(f'[Population {i:0>3d}]: {new_indv}')
        else:
            logger.info(f"Recover from {self.recovery}")
            with open(self.recovery) as f:
                data = json.loads(f.read())
            
            latest_iteration = data['iteration']
            population = [Individual(np.array(factors), ppl) for factors, ppl in data['population']]
            if "history" in data:
                self.history = [Individual(np.array(factors), ppl) for factors, ppl in data['history']]

        population_str = '\n'.join(map(str, population))
        logger.info(f"Iteration #{latest_iteration}\nPopulation:\n{population_str}")
        logger.info("Start Evolution")

        best_indv = Individual(None, np.inf)
        best_ppl_records = []

        with tqdm(
            total=self.max_time_budget,
            desc="Searching",
        ) as t:
            for i in range(latest_iteration, latest_iteration + self.max_time_budget):
                parents = sorted(population, key=lambda x: x.ppl)[:self.parents_size]
                self.log(i, parents)

                current_best_indv = parents[0]
                best_ppl_records.append(current_best_indv.ppl)
                t.set_postfix({"ppl": current_best_indv.ppl})

                logger.info(f"[Iter #{i + 1:0>3d} Best] {current_best_indv}")

                if current_best_indv.ppl < best_indv.ppl:
                    best_indv = current_best_indv

                population = parents

                # 变异
                for j in range(self.mutation_numbers):
                    idx = np.random.randint(self.parents_size)
                    mutated_indv = self.mutate(parents[idx])
                    population.append(mutated_indv)
                    logger.info(f'[Mutate #{i:0>3d} / #{j:0>3d}] {parents[idx]} / {mutated_indv}')

                # 交叉
                for j in range(self.crossover_size):
                    idx1, idx2 = np.random.choice(self.parents_size, 2, replace=False)
                    idx1, idx2 = sorted([idx1, idx2])
                    crossover_indv = self.crossover(parents[idx1], parents[idx2])
                    if crossover_indv is None:
                        logger.info(f'Crossover reach max {self.max_crossover_try} trys. Mutate from parent #1 instead.')
                        crossover_indv = self.mutate(parents[idx1])
                    logger.info(f'[Crossover #{i:0>3d} / #{j:0>3d}] {parents[idx1]} / {parents[idx2]} / {crossover_indv}')
                    population.append(crossover_indv)

                t.update(1)

                if i >= self.max_time_budget - 2:
                    population_str = '\n'.join(map(str, population))
                    logger.info(f'[Iter #{i + 1:0>3d} New Population]\n{population_str}')

                # early stop
                # if i >= latest_iteration + 20 and abs(best_ppl_records[i] - best_ppl_records[i-20]) < 0.001:
                #     break

        final_population = sorted(population, key=lambda x: x.ppl)[:self.parents_size]
        self.log(i, final_population)
        logger.info(f"PPL curve: {best_ppl_records}")
        return final_population[0].factors
