from typing import List, Callable

class ItemUtility:
    def __init__(self, item: str, sum_of_utility: float = 0, max_utility: float = 0):
        self.ITEM = item
        self.sum = sum_of_utility
        self.max = max_utility
        self.utilities = dict()
        self.is_depend = len(item) > 1
        self.length = 0

    def get_utility(self, transaction: str):
        if self.utilities.get(transaction):
            return self.utilities.get(transaction)
        return 0

    def set_utility(self, transaction: str, utility: float):
        self.utilities[transaction] = utility
        self.max = max(self.max, utility)
        self.sum += utility
        self.length += 1

    def __str__(self):
        return f"Item name: {self.ITEM}, sum: {self.sum}, max: {self.max}, utilities: {self.utilities}\n"

    def __repr__(self):
        return self.__str__()
    
    def __gt__(self, other: 'ItemUtility'):
        return self.sum > other.sum
    
    def __eq__(self, other):
        return isinstance(other, ItemUtility) and self.ITEM == other.ITEM

    def __hash__(self):
        return hash(self.ITEM)

class BayesianMiner:
    def __init__(self, database: list, top_k: int, min_sup: int = 0):
        self.DATABASE: List[dict] = database
        self.TOP_K: int = top_k
        self.min_sup: float = min_sup
        self.top_k_candidates: List[ItemUtility] = list()
        self.utility_dicts: dict[str, ItemUtility] = dict()
        self.dependence_list: List[ItemUtility] = list()
        self.independence_list: List[ItemUtility] = list()

    def __create_utility_lists(self):
        for transaction_id, transaction in enumerate(self.DATABASE):
            for item, utility in transaction.items():
                if item not in self.utility_dicts:
                    self.utility_dicts[item] = ItemUtility(item=item)
                self.utility_dicts[item].set_utility(transaction_id, utility) 
                
                if self.utility_dicts[item].is_depend:
                    self.dependence_list = self.__add_to_suitable_list(self.dependence_list, self.utility_dicts[item])
                else: 
                    self.independence_list = self.__add_to_suitable_list(self.independence_list, self.utility_dicts[item])


    def __add_to_suitable_list(self, chosen_list: List[ItemUtility], item_utility: ItemUtility):
        chosen_list.append(item_utility)
        return self.__clear_duplicate(chosen_list)

    def __clear_duplicate(self, input_list: List[ItemUtility]):
        return list(set(input_list))
    
    def __sort(self, input_list: List[ItemUtility], key_func: Callable[[ItemUtility], float], reverse: bool = True):
        return sorted(input_list, key=key_func, reverse=reverse)
    
    def __set_top_k(self, input_list: List[ItemUtility]):
        clear_duplicate_sorted_list = self.__sort(self.__clear_duplicate(input_list), key_func=lambda x: x.sum)

        return clear_duplicate_sorted_list[:self.TOP_K]

    def __set_top_k_candidates(self):
        self.top_k_candidates = self.__set_top_k(self.dependence_list + self.independence_list)
    
    def __set_min_sup(self):
        self.min_sup = self.top_k_candidates[-1].sum

    def __extract_item_names(self, utility_item: ItemUtility):
        return list(utility_item.ITEM)
    
    def __get_item_utility(self, name: str):
        return self.utility_dicts.get(name)

    def __get_extracted_item_utilities(self, depend: ItemUtility):
        item_names = self.__extract_item_names(depend)
        return list(map(lambda x: self.__get_item_utility(x), item_names))

    def __calculate_support_of_depends(self, chosen_list: List[ItemUtility]):
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
    
    def __find_top_k_bayesian_networks(self, item_utilities: List[ItemUtility]):
        for index, current in enumerate(item_utilities):
            next_item_utilities: List[ItemUtility] = list()
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

    def create_new_item_utility(self, old_item_1: ItemUtility, old_item_2: ItemUtility):
        tail_item_name = ''.join([ch for ch in old_item_2.ITEM if ch not in old_item_1.ITEM])
        reverse_tail_item_name = ''.join([ch for ch in old_item_1.ITEM if ch not in old_item_2.ITEM])
        
        if not tail_item_name or not reverse_tail_item_name:
            return None

        tail: ItemUtility = self.__get_item_utility(tail_item_name)
        
        new_item = ItemUtility(item=old_item_1.ITEM + tail_item_name)

        for transaction, utility in old_item_1.utilities.items():
            new_item.set_utility(transaction=transaction, utility=utility * tail.get_utility(transaction))
        
        return new_item

    def run(self):
        self.__create_utility_lists()
        self.dependence_list = self.__set_top_k(self.dependence_list)
        self.independence_list = self.__set_top_k(self.independence_list)
        self.__set_top_k_candidates()
        self.__set_min_sup()
        self.__calculate_support_of_depends(self.dependence_list)
        self.__find_top_k_bayesian_networks(self.top_k_candidates)


DATABASE = [
    {'a': 0.1, 'b': 0.3, 'cd': 0.5},
    {'a': 0.9, 'bc': 0.8, 'de': 0.6},
    {'ac': 0.7, 'be': 0.7, 'd': 0.9},
    {'ab': 0.6, 'c': 0.4, 'd': 0.1, 'e': 0.5},
    {'b': 0.8, 'd': 0.3, 'e': 0.2},
    {'cd': 0.3, 'e': 0.5}
]
TOP_K = 25

bayes_miner = BayesianMiner(DATABASE, TOP_K)
bayes_miner.run()
print(bayes_miner.get_top_k_candidates())
