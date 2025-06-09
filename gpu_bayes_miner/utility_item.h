#ifndef UTILITY_ITEM_CUH
#define UTILITY_ITEM_CUH

#include <string>
#include <vector>
#include <unordered_map>

struct Transaction {
    std::vector<std::string> items;
    std::vector<int> quantities;
    std::vector<int> profits;
    std::vector<double> probabilities;
};

struct UtilityTransaction {
    int utility;
    int remaining_utility;
    float support;
};

struct UtilityItem {
    std::string item;
    int sum_utility = 0;
    double sum_support = 0;
    int sum_remaining_utility = 0;
    int existance = 0;
    std::unordered_map<int, UtilityTransaction> utilities;

    void set_utility(int transaction_id, double probability, int utility, int remaining_utility) {
        UtilityTransaction utx;
        utx.utility = utility;
        utx.remaining_utility = remaining_utility;
        utx.support = static_cast<float>(probability);

        this->utilities[transaction_id] = utx;
        this->sum_utility += utility;
        this->sum_remaining_utility += remaining_utility;
        this->sum_support += probability;
        this->existance += 1;
    }

};

#endif
