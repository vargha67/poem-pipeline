#ifndef LIGHTHOUSE_H
#define LIGHTHOUSE_H

#include "Includes.h"
#include <map>
#include "StaticFunctions.h"

class Lighthouse
{
	vector<string> attributeNames;

	unordered_map<ind, string> keyToOutcome;
	unordered_map<string, ind> outcomeToKey;
	vector<unordered_map<ind, numm>> keyToAttribute;
	vector<unordered_map<numm, ind>> attributeToKey;

	unordered_map<vector<ind>, vector<ind>, ind_vector_hasher> tuples; // [Attributes] -> [outcomes1, outcome2, ..., totalcount, RCTIndex]

	ind D;
	ind N;
	ind O;
	ind totalN;

	vector<numm> lambdas; // R_j -> Lambda_j
	unordered_map<ind, vector<numm>> RCT; // region -> [models, rulescounts per outcome, tuplecounts per outcome, total Tuples Counts]

	vector<vector<ind>> patterns; // [Attributes, outcome, support, correctCount]
	bool match(vector<ind>& tuple, vector<ind>& pattern);
	bool patternsEqual(vector<ind>& pattern1, vector<ind>& pattern2);
	void addRule(vector<ind>& rule);
	void iterativeScaling();
	numm evalStraight(vector<ind>& pattern, vector<ind>& bestPattern, numm& bestGain);
	void generateSample(vector<vector<ind>>& sample, ind sampleSize);
	void addSimpleAncestors(vector<ind>& protoPattern, unordered_map<vector<ind>, vector<numm>, ind_vector_hasher>& candidates, ind depth, vector<ind>& data);
public:
	Lighthouse(string filename, vector<ind> dataColumns, ind outcomeColumn);
	~Lighthouse();

	numm KLDivergence();
	bool originalFlashlight(ind numberRules, ind sampleSize, numm minSupport, bool removeInactivatedPatterns, bool verbose);
	void printTable(numm baseKL, string fileName);
	string getTable();
	void LENS(ind numberRules, bool verbose, ind sampleSize, ind FLCandidates, ind mode);
};
#endif // !LIGHTHOUSE_H