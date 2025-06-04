class UtilityTransaction:
    def __init__(self, utility: int, probability: float, remaining_utility: int):
        self.utility = utility
        self.remaining_utility = remaining_utility
        self.probability = probability
    
    def __str__(self):
        return f"Utility: {self.utility}, Remaining utility: {self.remaining_utility}, Probability: {self.probability}\n"

    def __repr__(self):
        return self.__str__()

class UtilityItem:
    def __init__(self, item: str):
        self.ITEM = item
        self.sum = 0
        self.max = 0
        self.sum_prob = 0
        self.utilities: dict[UtilityTransaction] = dict()

    def get_utility(self, transaction: str):
        if self.utilities.get(transaction):
            return self.utilities.get(transaction)
        return 0

    def set_utility(self, transaction: int, probability: float, utility: int, remaining_utility: int):
        self.utilities[transaction] = UtilityTransaction(utility, probability, remaining_utility)
        self.max = max(self.max, utility)
        self.sum += utility
        self.sum_prob += probability

    def __str__(self):
        return f"Item name: {self.ITEM}, sum: {self.sum}, max: {self.max}, utilities: {self.utilities}, probability: {self.sum_prob}\n"

    def __repr__(self):
        return self.__str__()
    
    def __gt__(self, other: 'UtilityItem'):
        return self.sum > other.sum
    
    def __eq__(self, other):
        return isinstance(other, UtilityItem) and self.ITEM == other.ITEM

    def __hash__(self):
        return hash(self.ITEM)