class UtilityItem:
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
    
    def __gt__(self, other: 'UtilityItem'):
        return self.sum > other.sum
    
    def __eq__(self, other):
        return isinstance(other, UtilityItem) and self.ITEM == other.ITEM

    def __hash__(self):
        return hash(self.ITEM)