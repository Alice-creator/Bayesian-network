from utility_item import UtilityItem

def create_utility_dict(database: list, support_probability: float = 0, support_utility: float = 0):
    utilities: dict[tuple[str], UtilityItem] = dict()

    for transaction_id, transaction in enumerate(database):
        items: list = transaction.get("items")
        quantities: list = transaction.get("quantities")
        probabilities: list = transaction.get("probabilities")
        profits: list = transaction.get("profits")

        remaining_utility = 0

        for index in range(len(items) - 1, -1, -1):
            is_supported = 1 if '(' in items[index] else 0
            item_name = tuple([items[index]])
            if item_name not in utilities:
                utilities[item_name] = UtilityItem(item=item_name)
            item_utility = quantities[index] * profits[index] * (is_supported * support_utility + 1)
            item_probability = probabilities[index] * (support_probability * is_supported + 1)
            utilities[item_name].set_utility(transaction_id, item_probability, item_utility, remaining_utility)
            remaining_utility += item_utility
            
    return utilities

def get_number_of_transaction(database: list):
    return len(database)

def get_sum_utility_of_database(database: list):
    sum_of_utility = 0
    for transaction_id, transaction in enumerate(database):
        items: list = transaction.get("items")
        quantities: list = transaction.get("quantities")
        profits: list = transaction.get("profits")
        for index in range(len(items)):
            sum_of_utility += quantities[index] * profits[index]
    return sum_of_utility