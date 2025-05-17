class ItemUtility:
    def __init__(self, item: str, sum_of_utility: float, max_utility: float):
        self.ITEM = item
        self.sum = sum_of_utility
        self.max = max_utility
        self.utilities = dict()

    def get_utility(self, transaction: str):
        return self.utilities.get(transaction)

    def set_utility(self, transaction: str, utility: float):
        self.utilities.__setitem__(transaction, utility)

class BayesianMiner:
    def __init__(self, database: list, top_k: int, min_sup: int = 0):
        self.DATABASE = database
        self.TOP_K = top_k
        self.min_sup = min_sup
        self.top_k_candidates = list()

    def __create_utility_lists(self):
        return None

    def __set_top_k_depends(self):
        self.depends = list()

    def __set_top_k_not_depends(self):
        self.not_depends = list()

    def __set_top_k_items_based_on_sum(self):
        return None

    def __set_min_sup(self):
        self.min_sup = 0

    def __get_not_depends_from_dependence(self, depend_item):
        return None

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
        self.__set_top_k_items_based_on_sum()
        self.__set_min_sup()
        self.__calculate_support_of_depends()
        self.__find_top_k_bayesian_networks()


DATABASE = [
    {'a': 0.1, 'b': 0.3, 'cd': 0.5},
    {'a': 0.9, 'bc': 0.8, 'de': 0.6},
    {'ac': 0.7, 'be': 0.7, 'c': 0.9},
    {'ab': 0.6, 'c': 0.4, 'd': 0.1, 'e': 0.5},
    {'b': 0.8, 'd': 0.3, 'e': 0.2},
    {'cd': 0.3, 'e': 0.5}
]
TOP_K = 4

bayes_miner = BayesianMiner(DATABASE, TOP_K)
bayes_miner.run()
print(bayes_miner.get_top_k_candidates())