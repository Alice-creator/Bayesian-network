#include <unordered_map>
#include <string>
#include <vector>
#include "utility_item.h"

std::unordered_map<std::string, UtilityItem> create_utility_mapper(const std::vector<Transaction>& database) {
    std::unordered_map<std::string, UtilityItem> utilities;

    for (int tid = 0; tid < database.size(); ++tid) {
        const auto& transaction = database[tid];
        const std::vector<std::string>& items = transaction.items;
        const std::vector<int>& quantities = transaction.quantities;
        const std::vector<int>& profits = transaction.profits;
        const std::vector<double>& probabilities = transaction.probabilities;

        int remaining_utility = 0;
        for (size_t i = 0; i < quantities.size(); ++i) {
            remaining_utility += quantities[i] * profits[i];
        }

        for (size_t i = 0; i < items.size(); ++i) {
            std::string item = items[i];
            int item_utility = quantities[i] * profits[i];
            remaining_utility -= item_utility;

            if (utilities.find(item) == utilities.end()) {
                utilities[item] = UtilityItem{item};  // requires constructor
            }

            utilities[item].set_utility(tid, probabilities[i], item_utility, remaining_utility);
        }
    }

    return utilities;
}


int get_number_of_transaction(std::vector<Transaction> database) {
    return static_cast<int>(database.size());
}

int get_sum_utility_of_database(std::vector<Transaction> database) {
    int total_utility = 0;
    for (const auto& transaction : database) {
        for (size_t i = 0; i < transaction.profits.size(); ++i) {
            total_utility += transaction.profits[i] * transaction.quantities[i];
        }
    }
    return total_utility;
}
