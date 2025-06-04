from typing import List, Callable
from utility_item import UtilityItem
from helper import create_utility_dict

class BayesianMiner:
    def __init__(self, utility_dict: dict[UtilityItem], top_k: int, min_sup: int = 0):
        self.TOP_K: int = top_k
        self.min_sup: float = min_sup
        self.min_utility = 0
        self.utility_dicts: dict[str, UtilityItem] = utility_dict
        self.top_k_candidates: List[UtilityItem] = list()     

    def __add_to_suitable_list(self, chosen_list: List[UtilityItem], item_utility: UtilityItem):
        chosen_list.append(item_utility)
        return self.__clear_duplicate(chosen_list)

    def __clear_duplicate(self, input_list: List[UtilityItem]):
        return list(set(input_list))
    
    def __sort(self, input_list: List[UtilityItem], key_func: Callable[[UtilityItem], float], reverse: bool = True):
        return sorted(input_list, key=key_func, reverse=reverse)

    def __get_top_k_candidates(self, utility_list: List[UtilityItem]):
        return self.__sort(utility_list, key_func=lambda item: item.sum)[:self.TOP_K]
    
    def __set_min_utility(self):
        self.min_utility = self.top_k_candidates[-1].sum

    def __extract_item_names(self, utility_item: UtilityItem):
        return list(utility_item.ITEM)
    
    def __get_item_utility(self, name: str):
        return self.utility_dicts.get(name)

    def __get_extracted_item_utilities(self, depend: UtilityItem):
        item_names = self.__extract_item_names(depend)
        return list(map(lambda x: self.__get_item_utility(x), item_names))

    def __calculate_support_of_depends(self, chosen_list: List[UtilityItem]):
        for depend in chosen_list:
            item_utilities = self.__get_extracted_item_utilities(depend)
            sorted_item_utilities = self.__sort(item_utilities, key_func=lambda x: x.length, reverse=False)
            item_small, item_big = sorted_item_utilities[:2]
            max_sup =  depend.sum + item_small.sum * item_big.max

            if max_sup > self.min_sup:
                for transaction, utility in item_small.utilities.items():
                    if transaction in item_big.utilities:
                        self.utility_dicts[depend.ITEM].set_utility(transaction, utility + item_big.get_utility(transaction))
                self.dependence_list = self.__add_to_suitable_list(self.dependence_list, depend)
                if depend.sum > self.min_sup:
                    self.__set_top_k_candidates()
                    self.__set_min_sup()
    
    def __find_top_k_bayesian_networks(self, item_utilities: List[UtilityItem]):
        for index, current in enumerate(item_utilities):
            next_item_utilities: List[UtilityItem] = list()
            for next_current in item_utilities[index + 1:]:
                item_small, item_big = self.__sort([current, next_current], key_func=lambda x: x.length, reverse=False)
                max_sup = item_small.sum * item_big.max
                if max_sup > self.min_sup:
                    new_item = self.create_new_item_utility(item_small, item_big)
                    if new_item is None:
                        continue
                    self.utility_dicts[new_item.ITEM] = new_item
                    self.__add_to_suitable_list(self.dependence_list, new_item)
                    if new_item.sum > self.min_sup:
                        next_item_utilities.append(new_item)
                        self.__set_top_k_candidates()
                        self.__set_min_sup()
            self.__find_top_k_bayesian_networks(next_item_utilities)

    def get_top_k_candidates(self):
        return self.top_k_candidates

    def create_new_item_utility(self, old_item_1: UtilityItem, old_item_2: UtilityItem):
        tail_item_name = ''.join([ch for ch in old_item_2.ITEM if ch not in old_item_1.ITEM])
        reverse_tail_item_name = ''.join([ch for ch in old_item_1.ITEM if ch not in old_item_2.ITEM])
        
        if not tail_item_name or not reverse_tail_item_name:
            return None

        tail: UtilityItem = self.__get_item_utility(tail_item_name)
        
        new_item = UtilityItem(item=old_item_1.ITEM + tail_item_name)

        for transaction, utility in old_item_1.utilities.items():
            new_item.set_utility(transaction=transaction, utility=utility * tail.get_utility(transaction))
        
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
        # Find first top k candidates to expand
        self.top_k_candidates = self.__get_top_k_candidates(list(self.utility_dicts.values()))
        # Set first min utility
        self.__set_min_utility()
        self.__find_top_k_bayesian_networks(self.top_k_candidates)


DATABASE = [
    {
        "items": ["A", "B", "CD"],
        "quantities": [2, 1, 3],
        "profits": [6, 5, 9],
        "probabilities": [0.8, 0.75, 0.6]
    },
    {
        "items": ["A", "BC", "DE"],
        "quantities": [1, 2, 3],
        "profits": [5, 6, 7],
        "probabilities": [0.85, 0.68, 0.63]
    },
    {
        "items": ["AC", "BE"],
        "quantities": [1, 2],
        "profits": [4, 5],
        "probabilities": [0.72, 0.66]
    },
    {
        "items": ["AB", "C", "D", "E"],
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
        "items": ["CD", "E"],
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
TOP_K = 5

bayes_miner = BayesianMiner(create_utility_dict(DATABASE), TOP_K, 0.5)
bayes_miner.run()
print(bayes_miner.get_top_k_candidates())
