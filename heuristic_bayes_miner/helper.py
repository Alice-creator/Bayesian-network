from utility_item import UtilityItem

def create_utility_dict(database: list):
    utilities: dict[tuple[str], UtilityItem] = dict()

    for transaction_id, transaction in enumerate(database):
        items: list = transaction.get("items")
        quantities: list = transaction.get("quantities")
        probabilities: list = transaction.get("probabilities")
        profits: list = transaction.get("profits")

        remaining_utility = sum(q * p for q, p in zip(quantities, profits))

        for index in range(len(items)):
            item_name = tuple([items[index]])
            if item_name not in utilities:
                utilities[item_name] = UtilityItem(item=item_name)
            item_utility = quantities[index] * profits[index]
            remaining_utility -= item_utility
            utilities[item_name].set_utility(transaction_id, probabilities[index], item_utility, remaining_utility)
            
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