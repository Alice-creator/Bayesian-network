#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include "helper.h"
#include "utility_item.h"
#include <algorithm>

using namespace std;

__device__ bool is_same_prefix(UtilityItem itemset_1, UtilityItem itemset_2){
    return itemset_1.prefix == itemset_2.prefix;
}

__device__ UtilityItem create_new_item_utility(UtilityItem old_itemset_1, UtilityItem old_itemset_2){
    return old_itemset_1;
}

__global__ void generate_pairs(vector<UtilityItem>* expandable_itemset, vector<UtilityItem> new_expandable_itemset) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx + 1; i < expandable_itemset->size(); i++) {
        if((*expandable_itemset)[i] == '\0') continue;
        else if (is_same_prefix((*expandable_itemset)[i], (*expandable_itemset)[idx]))
        {
            UtilityItem old_itemset_1 = (*expandable_itemset)[idx];
            UtilityItem old_itemset_2 = (*expandable_itemset)[i];
            UtilityItem new_item_utility = create_new_item_utility(old_itemset_1, old_itemset_2);
            new_expandable_itemset.push_back(new_item_utility);
        }
        
    }
}

struct BayesianMiner {
    const int TOP_K;
    const int NUMBER_OF_TRANSACTIONS;
    const int DATABASE_UTILITY;
    const double MIN_SUPPORT;
    vector<UtilityItem> candidates;
    vector<UtilityItem> expandable_itemsets;
    int min_utility = 0;
    unordered_map<string, UtilityItem> utility_map;

    BayesianMiner(int k, int num_tx, int db_util, double min_sup)
        : TOP_K(k), NUMBER_OF_TRANSACTIONS(num_tx), DATABASE_UTILITY(db_util), MIN_SUPPORT(min_sup) {}

    unordered_map<string, UtilityItem> get_utility_map(){
        return utility_map;
    }

    void run(){
        this->utility_map = this->get_valid_min_support_candidates(this->utility_map);
        this->expandable_itemsets = this->get_expandable(this->utility_map, this->min_utility);
        this->candidates = this->get_top_k_candidates(this->expandable_itemsets, this->TOP_K);
        this->set_min_utility(this->candidates);
    }

    private:
        unordered_map<string, UtilityItem> get_valid_min_support_candidates(unordered_map<string, UtilityItem> utility_map){
            unordered_map<string, UtilityItem> result;
            for(auto &utility_item: utility_map){
                if(utility_item.second.sum_support >= this->MIN_SUPPORT){
                    result[utility_item.first] = utility_item.second;
                }
            }
            return result;
        }

        vector<UtilityItem> get_expandable(unordered_map<string, UtilityItem> utility_map, int min_utility){
            vector<UtilityItem> result;
            for(auto &utility_item: utility_map){
                if(utility_item.second.sum_remaining_utility + utility_item.second.sum_utility >= min_utility){
                    result.push_back(utility_item.second);
                }
            }
            return result;
        }

        vector<UtilityItem> get_top_k_candidates(vector<UtilityItem> expandable_itemset, int top_k) {
            // Sort the input list, not utility_map
            sort(expandable_itemset.begin(), expandable_itemset.end(), [](const UtilityItem& a, const UtilityItem& b) {
                return a.sum_utility > b.sum_utility;
            });

            if (expandable_itemset.size() > top_k) {
                expandable_itemset.resize(top_k);
            }

            return expandable_itemset;
        }

        void set_min_utility(const std::vector<UtilityItem>& top_k_candidates) {
            if (!top_k_candidates.empty()) {
                this->min_utility = top_k_candidates.back().sum_utility;
            } else {
                this->min_utility = 0;
            }
        }
};


int main(){

        vector<Transaction> DATABASE = {
        {
            {"A", "B", "(CD)"},
            {2, 1, 3},
            {6, 5, 9},
            {0.8, 0.75, 0.6}
        },
        {
            {"A", "(BC)", "(DE)"},
            {1, 2, 3},
            {5, 6, 7},
            {0.85, 0.68, 0.63}
        },
        {
            {"(AC)", "(BE)"},
            {1, 2},
            {4, 5},
            {0.72, 0.66}
        },
        {
            {"(AB)", "C", "D", "E"},
            {2, 1, 2, 1},
            {7, 3, 4, 2},
            {0.78, 0.7, 0.6, 0.65}
        },
        {
            {"B", "C", "D", "E"},
            {2, 1, 2, 1},
            {6, 3, 5, 4},
            {0.75, 0.66, 0.59, 0.61}
        },
        {
            {"(CD)", "E"},
            {2, 1},
            {6, 3},
            {0.64, 0.67}
        },
        {
            {"A", "B", "C", "D", "E"},
            {2, 2, 1, 2, 1},
            {6, 5, 4, 3, 2},
            {0.85, 0.7, 0.65, 0.6, 0.68}
        }
    };
    const int TOP_K = 10;
    const double MIN_SUPPORT = 0.5;
    BayesianMiner bayesian_miner(TOP_K, get_number_of_transaction(DATABASE), get_sumutility_of_database(DATABASE), MIN_SUPPORT);
    bayesian_miner.utility_map = create_utility_mapper(DATABASE);
    bayesian_miner.run();
    unordered_map<string, UtilityItem> util_map = bayesian_miner.get_utility_map();
    for (const auto& pair : util_map) {
        std::cout << "Key: " << pair.first << std::endl;
        pair.second.print();
        std::cout << "------------------------" << std::endl;
    }
    // std::cout << "\nTop-K Candidates:\n";
    // for (const auto& item : bayesian_miner.candidates) {
    //     item.print();
    //     std::cout << "------------------------\n";
    // }

    // std::cout << "\nExpandable Itemsets:\n";
    // for (const auto& item : bayesian_miner.expandable_itemsets) {
    //     item.print();
    //     std::cout << "------------------------\n";
    // }
    return 0;
}