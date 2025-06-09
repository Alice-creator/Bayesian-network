#ifndef HELPER_H
#define HELPER_H

#include <unordered_map>
#include <string>
#include <vector>
#include <utility_item.h>

std::unordered_map<std::string, int> create_utility_mapper(std::vector<Transaction> database);

int get_number_of_transaction(std::vector<Transaction> database);

int get_sumutility_of_database(std::vector<Transaction> database);

#endif
