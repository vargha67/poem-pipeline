#ifndef STATIC_FUNCTIONS_H
#define STATIC_FUNCTIONS_H

#include "Includes.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <map>
#include <set>

class string_vector_hasher
{
public:
	size_t operator()(vector<string> const &vec) const
	{
		size_t seed = vec.size();
		for (auto &i : vec)
		{
			seed ^= hash<string>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

class ind_vector_hasher
{
public:
	std::size_t operator()(std::vector<ind> const &vec) const
	{
		std::size_t seed = vec.size();
		for (auto &i : vec)
		{
			seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

#endif