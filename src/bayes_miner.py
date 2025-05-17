from typing import List

class ItemUtility:
    def __init__(self, item: str, sum_of_utility: float = 0, max_utility: float = 0):
        self.ITEM = item
        self.sum = sum_of_utility
        self.max = max_utility
        self.utilities = dict()
        self.is_depend = len(item) > 1

    def get_utility(self, transaction: str):
        return self.utilities.get(transaction)

    def set_utility(self, transaction: str, utility: float):
        self.utilities[transaction] = utility
        self.max = max(self.max, utility)
        self.sum += utility

    def __str__(self):
        return f"Item name: {self.ITEM}, sum: {self.sum}, max: {self.max}, utilities: {self.utilities}\n"

    def __repr__(self):
        return self.__str__()
    
    def __gt__(self, other: 'ItemUtility'):
        return self.sum > other.sum

class BayesianMiner:
    def __init__(self, database: list, top_k: int, min_sup: int = 0):
        self.DATABASE: List[dict] = database
        self.TOP_K: int = top_k
        self.min_sup: float = min_sup
        self.top_k_candidates: List[ItemUtility] = list()
        self.utility_lists: dict[str, ItemUtility] = dict()
        self.dependence_list: List[ItemUtility] = list()
        self.independence_list: List[ItemUtility] = list()

    def __create_utility_lists(self):
        for transaction_id, transaction in enumerate(self.DATABASE):
            for item, utility in transaction.items():
                if item not in self.utility_lists:
                    self.utility_lists[item] = ItemUtility(item=item)
                self.utility_lists[item].set_utility(transaction_id, utility) 
                
                if self.utility_lists[item].is_depend:
                    self.dependence_list = self.__add_to_suitable_list(self.dependence_list, self.utility_lists[item])
                else: 
                    self.independence_list = self.__add_to_suitable_list(self.independence_list, self.utility_lists[item])


    def __add_to_suitable_list(self, chosen_list: List[ItemUtility], item_utility: ItemUtility):
        chosen_list.append(item_utility)
        return self.__clear_duplicate(chosen_list)

    def __clear_duplicate(self, input_list: List[ItemUtility]):
        return list(set(input_list))
    
    def __sort(self, input_list: List[ItemUtility]):
        return sorted(input_list, key=lambda x: x.sum, reverse=True)
    
    def __set_top_k(self, input_list: List[ItemUtility]):
        sorted_list = self.__sort(input_list)
        return sorted_list[:self.TOP_K]

    def __set_top_k_candidates(self):
        self.top_k_candidates = self.__set_top_k(self.dependence_list + self.independence_list)
    
    def __set_min_sup(self):
        self.min_sup = self.top_k_candidates[-1].sum

    def __calculate_support_of_depends(self):
        return None

    def __find_top_k_bayesian_networks(self):
        return None

    def get_top_k_candidates(self):
        return self.top_k_candidates

    def create_new_item_utility(self, old_item_1: ItemUtility, old_item_2: ItemUtility):
        return None

    def run(self):
        self.__create_utility_lists()
        self.dependence_list = self.__set_top_k(self.dependence_list)
        self.independence_list = self.__set_top_k(self.independence_list)
        self.__set_top_k_candidates()
        self.__set_min_sup()
        # self.__calculate_support_of_depends()
        # self.__find_top_k_bayesian_networks()


DATABASE = [
    {'a': 0.1, 'b': 0.3, 'cd': 0.5},
    {'a': 0.9, 'bc': 0.8, 'de': 0.6},
    {'ac': 0.7, 'be': 0.7, 'd': 0.9},
    {'ab': 0.6, 'c': 0.4, 'd': 0.1, 'e': 0.5},
    {'b': 0.8, 'd': 0.3, 'e': 0.2},
    {'cd': 0.3, 'e': 0.5}
]
TOP_K = 5

bayes_miner = BayesianMiner(DATABASE, TOP_K)
bayes_miner.run()
