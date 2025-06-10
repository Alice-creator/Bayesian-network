#ifndef HELPER_H
#define HELPER_H

#include <unordered_map>
#include <string>
#include <vector>
#include "utility_item.h"

std::unordered_map<std::string, UtilityItem> create_utility_mapper(const std::vector<Transaction>& database);
int get_number_of_transaction(const std::vector<Transaction> database);
int get_sumutility_of_database(const std::vector<Transaction> database);

#endif
