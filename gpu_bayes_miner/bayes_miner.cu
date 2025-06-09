#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <helper.h>
#include <utility_item.h>

using namespace std;

struct BayesianMiner {
    const int TOP_K;
    const int NUMBER_OF_TRANSACTIONS;
    const int DATABASE_UTILITY;
    const double MIN_SUPPORT;
    int min_utility = 0;
    unordered_map<string, UtilityItem> utility_map;

    BayesianMiner(int k, int num_tx, int db_util, double min_sup)
        : TOP_K(k), NUMBER_OF_TRANSACTIONS(num_tx), DATABASE_UTILITY(db_util), MIN_SUPPORT(min_sup) {}
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
    return 0;
}