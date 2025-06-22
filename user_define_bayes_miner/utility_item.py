import math

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
    def __init__(self, item: tuple[str]):
        self.ITEM = item
        self.sum_utility = 0
        self.sum_prob = 0
        self.sum_ru = 0
        self.existance = 0
        self.max_prob = 0
        self.utilities: dict[int, UtilityTransaction] = dict()

    def get_probability(self, id: int):
        if self.utilities.get(id):
            return self.utilities.get(id).probability
        return 0
    
    def get_utility(self, id: int):
        if self.utilities.get(id):
            return self.utilities.get(id).utility
        return 0
    
    def get_remaining(self, id: int):
        if self.utilities.get(id):
            return self.utilities.get(id).remaining_utility
        return 0
    
    def set_utility(self, transaction: int, probability: float, utility: int, remaining_utility: int):
        if not math.isclose(probability, 0.0, abs_tol=1e-9):
            self.utilities[transaction] = UtilityTransaction(utility, probability, remaining_utility)
            self.sum_utility += utility
            self.sum_ru += remaining_utility
            self.sum_prob += probability
            self.max_prob = max(self.max_prob, probability)
            self.existance += 1

    def __str__(self):
        return f"Item name: {self.ITEM}, sum: {self.sum_utility}, probability: {self.sum_prob}, utilities: {self.utilities}\n"

    def __repr__(self):
        return self.__str__()
    
    def __gt__(self, other: 'UtilityItem'):
        return self.sum_utility > other.sum_utility
    
    def __eq__(self, other):
        return isinstance(other, UtilityItem) and self.ITEM == other.ITEM

    def __hash__(self):
        return hash(self.ITEM)