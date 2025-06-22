from typing import List, Callable
from utility_item import UtilityItem
from helper import create_utility_dict

class BayesianMiner:
    def __init__(self, utility_dict: dict[UtilityItem], top_k: int, min_sup: int = 0):
        self.TOP_K: int = top_k
        self.min_sup: float = min_sup
        self.min_utility = 0
        self.utility_dicts: dict[tuple[str], UtilityItem] = utility_dict
        self.top_k_candidates: List[UtilityItem] = list()
        self.exandable_itemset: List[UtilityItem] = list() 

    def __sort(self, input_list: List[UtilityItem], key_func: Callable[[UtilityItem], float], reverse: bool = True):
        return sorted(input_list, key=key_func, reverse=reverse)

    def __get_top_k_candidates(self, utility_list: List[UtilityItem]):
        return self.__sort(utility_list, key_func=lambda item: item.sum_utility)[:self.TOP_K]
    
    def __get_expandable(self, utility_list: List[UtilityItem], min_utility):
        return list(filter(lambda item: item.sum_utility + item.sum_ru > min_utility, utility_list))

    def __set_min_utility(self):
        self.min_utility = self.top_k_candidates[-1].sum_utility

    def __get_item_utility(self, name: tuple[str]):
        return self.utility_dicts.get(name)

    def __is_able_to_combine(self, item1: UtilityItem, item2: UtilityItem):
        return not set(item1.utilities.keys()).isdisjoint(item2.utilities.keys())

    
    def __find_top_k_bayesian_networks(self, item_utilities: List[UtilityItem]):
        for index, current in enumerate(item_utilities):
            if not self.__is_promising(current):
                continue

            next_item_utilities = []

            for next_current in item_utilities[index + 1:]:
                if not self.__is_able_to_combine(current, next_current):
                    continue

                new_item = self.__try_combine(current, next_current)
                if new_item is None:
                    continue

                next_item_utilities.append(new_item)
                self.__process_new_item(new_item)

            self.__find_top_k_bayesian_networks(next_item_utilities)

    # --- Helper methods ---

    def __is_promising(self, item: UtilityItem) -> bool:
        return item.sum_utility + item.sum_ru >= self.min_utility

    def __try_combine(self, item1: UtilityItem, item2: UtilityItem):
        item_small, item_big = self.__sort([item1, item2], key_func=lambda x: len(x.utilities), reverse=False)
        new_item = self.__create_new_item_utility(item_small, item_big)
        if new_item and new_item.sum_prob > self.min_sup:
            return new_item
        return None

    def __process_new_item(self, item: UtilityItem):
        self.utility_dicts[item.ITEM] = item
        if item.sum_utility > self.min_utility:
            self.top_k_candidates.append(item)
            self.top_k_candidates = self.__get_top_k_candidates(self.top_k_candidates)
            self.__set_min_utility()


    def get_top_k_candidates(self):
        return self.top_k_candidates

    def __create_new_item_utility(self, old_item_1: UtilityItem, old_item_2: UtilityItem):
        item1_set = set(old_item_1.ITEM)
        tail_item = tuple(item for item in old_item_2.ITEM if item not in item1_set)
        if not tail_item:
            return None

        tail: UtilityItem = self.__get_item_utility(tail_item)
        new_item = UtilityItem(item=tuple(old_item_1.ITEM + tail.ITEM))

        for id, transaction in old_item_1.utilities.items():
            new_item.set_utility(transaction=id, probability=transaction.probability * tail.get_probability(id), utility=transaction.utility + tail.get_utility(id), remaining_utility=min(transaction.remaining_utility, tail.get_remaining(id)))
        return new_item

    def __get_valid_min_support_candidates(self, utility_dict: dict[str, UtilityItem]):
        return {
            name: item
            for name, item in utility_dict.items()
            if item.sum_prob >= self.min_sup
        }

    def run(self):
        # Remove candidates where: candidate.prob < min support
        self.utility_dicts = self.__get_valid_min_support_candidates(self.utility_dicts)
        # Find expandable itemset to expand, first top k candidate to return
        self.exandable_itemset = self.__get_expandable(list(self.utility_dicts.values()), self.min_utility)
        self.top_k_candidates = self.__get_top_k_candidates(list(self.utility_dicts.values()))
        # Set first min utility
        self.__set_min_utility()
        # Mine top K candidates
        self.__find_top_k_bayesian_networks(self.exandable_itemset)


DATABASE = [
    {
        "items": ["A", "B", "(CD)"],
        "quantities": [2, 1, 3],
        "profits": [6, 5, 9],
        "probabilities": [0.8, 0.75, 0.6]
    },
    {
        "items": ["A", "(BC)", "(DE)"],
        "quantities": [1, 2, 3],
        "profits": [5, 6, 7],
        "probabilities": [0.85, 0.68, 0.63]
    },
    {
        "items": ["(AC)", "(BE)"],
        "quantities": [1, 2],
        "profits": [4, 5],
        "probabilities": [0.72, 0.66]
    },
    {
        "items": ["(AB)", "C", "D", "E"],
        "quantities": [2, 1, 2, 1],
        "profits": [7, 3, 4, 2],
        "probabilities": [0.78, 0.7, 0.6, 0.65]
    },
    {
        "items": ["B", "C", "D", "E"],
        "quantities": [2, 1, 2, 1],
        "profits": [6, 3, 5, 4],
        "probabilities": [0.75, 0.66, 0.59, 0.61]
    },
    {
        "items": ["(CD)", "E"],
        "quantities": [2, 1],
        "profits": [6, 3],
        "probabilities": [0.64, 0.67]
    },
    {
        "items": ["A", "B", "C", "D", "E"],
        "quantities": [2, 2, 1, 2, 1],
        "profits": [6, 5, 4, 3, 2],
        "probabilities": [0.85, 0.7, 0.65, 0.6, 0.68]
    }
]
TOP_K = 10

bayes_miner = BayesianMiner(create_utility_dict(DATABASE), TOP_K, 0.5)
bayes_miner.run()
print(bayes_miner.get_top_k_candidates())
