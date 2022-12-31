#include "Lighthouse.h"
#include <fstream>
#include <sstream>
#include <deque>

Lighthouse::Lighthouse(string filename, vector<ind> dataColumns, ind outcomeColumn) // parse the data
{
	// build dictionaries for attributes and fill them afterwards
	// here the data is transformed to multisets such that for each combination of eplanatory attributes the tuples per outcome as well as total count is stored
	vector<unordered_set<numm>> attributeValues(dataColumns.size(), unordered_set<numm>());
	keyToAttribute.resize(dataColumns.size());
	attributeToKey.resize(dataColumns.size());
	attributeNames.resize(dataColumns.size() + 1);

	vector<vector<string>> raw(0);
	string line;
	ifstream data(filename);
	while (getline(data, line)) {
		stringstream lineStream(line);
		string cell;
		vector<string> lineVec;
		while (getline(lineStream, cell, ',')) { lineVec.push_back(cell); }
		raw.push_back(lineVec);
	}
	for (ind j = 0; j < dataColumns.size(); ++j) {
		attributeNames[j] = raw[0][dataColumns[j]];
	}
	attributeNames[dataColumns.size()] = raw[0][outcomeColumn];
	for (ind i = 1; i < raw.size(); ++i) {
		string outcome = raw[i][outcomeColumn];
		if (outcomeToKey.count(outcome) == 0) {
			outcomeToKey.emplace(outcome, outcomeToKey.size());
			keyToOutcome.emplace(keyToOutcome.size(), outcome);
		}
		for (ind j = 0; j < dataColumns.size(); ++j) {
			numm attribute(stof(raw[i][dataColumns[j]]));
			if (attributeValues[j].count(attribute) == 0) {
				attributeValues[j].insert(attribute);
			}
		}
	}
	for (ind j = 0; j < dataColumns.size(); ++j) {
		vector<numm> values(attributeValues[j].begin(), attributeValues[j].end());
		std::sort(values.begin(), values.end());
		for (ind i = 0; i < values.size(); ++i) {
			keyToAttribute[j].emplace(i, values[i]);
			attributeToKey[j].emplace(values[i], i);
		}
	}
	for (ind i = 1; i < raw.size(); ++i) {
		vector<ind> tuple(dataColumns.size(), 0);
		for (ind j = 0; j < dataColumns.size(); ++j) {
			tuple[j] = attributeToKey[j].find(stof(raw[i][dataColumns[j]]))->second;
		}
		auto ref = tuples.find(tuple);
		if (ref == tuples.end()) { ref = tuples.emplace(tuple, vector<ind>(keyToOutcome.size() + 2, 0)).first; }
		ref->second[keyToOutcome.size()] += 1;
		ref->second[outcomeToKey.find(raw[i][outcomeColumn])->second] += 1;
	}
	N = tuples.size(); // number of tuples total
	D = dataColumns.size(); // number of dimensions
	O = outcomeToKey.size(); // number of outcome values
	vector<ind> resultcounts(O, 0);
	totalN = 0;
	for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
		for (ind o = 0; o < O; ++o) {
			resultcounts[o] += iter->second[o];
		}
		totalN += iter->second[O];
	}
	// construct RCT for with default case as no rule exists yet
	vector<numm> rctBase(3 * O + 1);
	for (ind o = 0; o < O; ++o) {
		rctBase[o] = 1.0 / (numm)O;
		rctBase[O + o] = 0;
		rctBase[2 * O + o] = resultcounts[o];
	}
	rctBase[3 * O] = totalN;
	RCT.emplace(0, rctBase);
	lambdas = vector<numm>(0); // R_j -> Lambda_j
	vector<ind> pattern(2 * D + 3, 0);
	for (ind d = 0; d < D; ++d) {
		pattern[D + d] = attributeToKey[d].size() - 1;
	}
	pattern[2 * D + 1] = totalN;
	for (ind o = 0; o < O; ++o) { // Add default rules for each outcome, i.e. one rule matching all attrbiutes per outcome value
		vector<ind> p2 = pattern;
		p2[2 * D] = o;
		p2[2 * D + 2] = resultcounts[o];
		addRule(p2);
	}
	iterativeScaling();
}

bool Lighthouse::match(vector<ind>& tuple, vector<ind>& pattern) // test whether a tupple matches a pattern
{
	for (ind i = 0; i < D; ++i) {
		if (pattern[i] > tuple[i] || pattern[D+i] < tuple[i]) { return false; }
	}
	return true;
}

bool Lighthouse::patternsEqual(vector<ind>& pattern1, vector<ind>& pattern2) // test whether two patterns are equal
{
	for (ind i = 0; i < D; ++i) {
		if (pattern1[i] != pattern2[i] || pattern1[D+i] != pattern2[D+i]) { return false; }
	}
	if (pattern1[D] != pattern2[D]) { return false; }
	return true;
}

void Lighthouse::addRule(vector<ind>& rule) // insert a rule into the Explanation table
{
	for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
		vector<ind> alocTup = iter->first; // for each affected pattern, the RCT has to be corrected
		if (match(alocTup, rule)) {
			ind oldIndex = iter->second[O + 1];
			ind newIndex = oldIndex + ((ind)1 << patterns.size());
			auto oldRef = RCT.find(oldIndex);
			auto newRef = RCT.find(newIndex);
			if (newRef == RCT.end()) { newRef = RCT.emplace(newIndex, vector<numm>(3 * O + 1, 0.0)).first; } // if new RCT entry doesn't exist yet, add it
			newRef->second[3 * O] += iter->second[O];
			oldRef->second[3 * O] -= iter->second[O];
			for (ind o = 0; o < O; ++o) {
				newRef->second[o] = oldRef->second[o];
				newRef->second[O + o] = oldRef->second[O + o];
				if (o == rule[2*D]) { newRef->second[O + o] += 1; }
				newRef->second[2 * O + o] += iter->second[o];
				oldRef->second[2 * O + o] -= iter->second[o];

			}
			iter->second[O + 1] = newIndex;
			if (oldRef->second[3 * O] <= 0) { RCT.erase(oldRef); } // if old RCT entry is empty now, clean it up
		}
	}
	patterns.push_back(rule);
	lambdas.push_back(0);
}

void Lighthouse::iterativeScaling() // perform iterative scaling as described in the paper
{
	numm limit = 0.1 / (numm)totalN;
	bool divergent = true;
	while (divergent) {
		divergent = false;
		for (ind p = 0; p < patterns.size(); ++p) {
			ind outcome = patterns[p][2*D];
			numm percentageApplicableTuples = (numm)patterns[p][2*D + 2] / (numm)totalN;
			numm expectedSum = 0;
			numm deriv = 0;
			numm delta = 0;
			for (auto iter = RCT.begin(); iter != RCT.end(); ++iter) {
				if (iter->first & ((ind)1 << p)) {
					numm term = (iter->second[3 * O] / (numm)totalN) * iter->second[outcome] * exp(delta * iter->second[O + outcome]);
					expectedSum += term;
					deriv += iter->second[O + outcome] * term;
				}
			}
			while (abs(expectedSum - percentageApplicableTuples) > limit && abs(delta) < 50) {
				divergent = true;
				if (abs((expectedSum - percentageApplicableTuples) / deriv) > 100) {
					deriv*=100;
				}
				delta -= (expectedSum - percentageApplicableTuples) / deriv;
				expectedSum = 0;
				deriv = 0;
				for (auto iter = RCT.begin(); iter != RCT.end(); ++iter) {
					if (iter->first & ((ind)1 << p)) {
						numm term = (iter->second[3 * O] / (numm)totalN) * iter->second[outcome] * exp(delta * iter->second[O + outcome]);
						expectedSum += term;
						deriv += iter->second[O + outcome] * term;
					}
				}
			}
			lambdas[p] += delta;
			if (delta != 0) {
				for (auto iter = RCT.begin(); iter != RCT.end(); ++iter) {
					if (iter->first & ((ind)1 << p)) {
						numm denominator = 0;
						vector<numm> numerator(O, 0);
						for (ind o = 0; o < O; ++o) {
							numm lambdaSum = 0;
							for (ind p2 = 0; p2 < patterns.size(); ++p2) {
								if ((patterns[p2][2*D] == o) && (iter->first & ((ind)1 << p2))) {
									lambdaSum += lambdas[p2];
								}
							}
							numerator[o] = exp(lambdaSum);
							denominator += exp(lambdaSum);
						}
						for (ind o = 0; o < O; ++o) {
							iter->second[o] = numerator[o] / denominator;
						}
					}
				}
			}
		}
	}
}

numm Lighthouse::KLDivergence() // measure KL divergence of current Explanation Table to empiricial distribution
{
	numm sum = 0;
	numm eps = 1e-15;
	for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
		numm total = (numm)iter->second[O];
		for (ind o = 0; o < O; ++o) {
			//numm pi = (numm)iter->second[o] / (numm)iter->second[O];
			//numm qi = RCT.find(iter->second[O + 1])->second[o];
			//pi = pi + eps;
			//qi = qi + eps;
			//if ((pi != 0) && (qi != 0)) { sum += iter->second[O] * (pi*log(pi / qi)); }

			// alternative method of computation to compare with IDS and decision trees: 
			numm qi = RCT.find(iter->second[O + 1])->second[o] + eps;
			numm pi = 1 + eps;
			numm cnt = (numm)iter->second[o];
			sum += (cnt * (pi * log(pi / qi))); 

			pi = eps;
			cnt = total - cnt;
			sum += (cnt * (pi * log(pi / qi)));
		}
	}
	return sum;
}

numm Lighthouse::evalStraight(vector<ind>& pattern, vector<ind>& bestPattern, numm& bestGain) // selects best outcome for pattern and stores it in bestPattern
{
	numm thisGain = 0;
	ind sup = 0;
	vector<ind> hits(O, 0);
	vector<numm> model(O, 0);
	for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
		vector<ind> tuple = iter->first;
		if (match(tuple, pattern)) {
			sup += iter->second[O];
			for (ind o = 0; o < O; ++o) {
				hits[o] += iter->second[o];
				model[o] += iter->second[O] * RCT.find(iter->second[O + 1])->second[o];
			}
		}

	}
	for (ind o = 0; o < O; ++o) {
		numm supportF = sup;
		numm trueRate = (numm)hits[o] / supportF;
		numm expRate = model[o] / supportF;
		if ((trueRate != 0) && (expRate != 0)) {
			numm gain = supportF * trueRate *log(trueRate / expRate);
			for (ind o2 = 0; o2 < O; ++o2) {
				if (o2 != o) {
					numm tr = (numm)hits[o2] / supportF;
					numm oldER = model[o2] / supportF;
					numm newER = (oldER / (1.0 - expRate)) * (1.0 - trueRate);
					if ((tr != 0) && (oldER != 0) && (newER != 0)) {
						gain += supportF * tr * (log(tr / oldER) - log(tr / newER));
					}
				}
			}
			thisGain = max(thisGain, gain);
			if (gain > bestGain) {
				bestGain = gain;
				bestPattern = pattern;
				bestPattern.push_back(o);
				bestPattern.push_back(sup);
				bestPattern.push_back(hits[o]);
			}
		}
	}
	return thisGain;
}


void Lighthouse::generateSample(vector<vector<ind>>& sample, ind sampleSize) { // draw a sample of desired size from the data weighted by total occurence
	// Because we transformed teh data to a multiset, naive sample from "tuples" neglects more common items and is not a fair sample from the original data.
	// So we generate number between 0 and N and accumulate the tuple count in order until the cumulative um reaches the randum count, this yields a fair randomly drawn tuple
	ind realSampleSize = min(totalN, sampleSize);
	random_device rd;
	mt19937_64 gen(rd());
	vector<ind> thresholds(totalN);
	for (ind i = 0; i < totalN; ++i) { thresholds[i] = i; }
	for (ind i = 0; i < realSampleSize; ++i) {
		uniform_int_distribution<ind> indexGen(i, totalN - 1);
		ind index = indexGen(gen);
		swap(thresholds[i], thresholds[index]);
	}
	thresholds.resize(realSampleSize);
	sort(thresholds.begin(), thresholds.end());
	ind s = 0;
	ind cumulative = 0;
	for (auto iter = tuples.begin(); iter != tuples.end() && s < realSampleSize; ++iter) {
		cumulative += iter->second[O];
		if (cumulative >= thresholds[s]) {
			sample[s] = iter->first;
			++s;
		}
	}
	sample.resize(s);
}

void Lighthouse::addSimpleAncestors(vector<ind>& protoPattern, unordered_map<vector<ind>, vector<numm>, ind_vector_hasher>& candidates, ind depth, vector<ind>& data)
{
	// generate ancestor patterns and att them to the candidate list (or update their counts if they already exist)
	if (depth < D) {
		if (protoPattern[depth] == protoPattern[D + depth]) {
			ind temp = protoPattern[depth];
			protoPattern[depth] = 0;
			protoPattern[D + depth] = attributeToKey[depth].size() - 1;
			addSimpleAncestors(protoPattern, candidates, depth + 1, data);
			protoPattern[depth] = temp;
			protoPattern[D + depth] = temp;
		}
		addSimpleAncestors(protoPattern, candidates, depth + 1, data);
	}
	else {
		auto ref = candidates.find(protoPattern);
		if (ref == candidates.end()) {
			ref = candidates.emplace(protoPattern, vector<numm>(2 * O + 1, 0)).first;
		}
		for (ind o = 0; o < O; ++o) {
			ref->second[2 * o] += data[o];
			ref->second[2 * o + 1] += (numm)data[O] * RCT.find(data[O + 1])->second[o];
		}
		ref->second[2 * O] += data[O];
	}
}

Lighthouse::~Lighthouse()
{
}

bool Lighthouse::originalFlashlight(ind numberRules, ind sampleSize, numm minSupport, bool removeInactivatedPatterns, bool verbose)
{
	// internal counting gain and time for verbose mode
	auto globalStart = chrono::high_resolution_clock::now();
	numm baseKL = KLDivergence();
	numm recentKL = baseKL;
	for (ind rules = 0; rules < numberRules; ++rules) {
		auto start = chrono::high_resolution_clock::now();

		// draw sample
		vector<vector<ind>> sample(sampleSize);
		generateSample(sample, sampleSize);

		// generate LCA table immediatly with aggregates [cf. Table 7 in El Gebaly et al. 2014]
		unordered_map<vector<ind>, vector<numm>, ind_vector_hasher> patternCandidates; // Format of map: [AttributesLow, AttributesHigh] -> [Out1Count, Out1Exp, Out2Count, ..., OutxExp, TotalCount]
		for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
			for (ind s = 0; s < sample.size(); ++s) {
				vector<ind> protoPattern(2 * D, 0);
				for (ind k = 0; k < D; ++k) {
					if (iter->first[k] == sample[s][k]) {
						protoPattern[k] = sample[s][k];
						protoPattern[D + k] = sample[s][k];
					}
					else {
						protoPattern[k] = 0;
						protoPattern[D + k] = attributeToKey[k].size() - 1;
					}
				}
				addSimpleAncestors(protoPattern, patternCandidates, 0, iter->second);
			}
		}
		// evalaute gain of ancestor patterns, i.e., correct aggregates cf. [Table 8 in El Gebaly et al. 2014] and [Sec. 3.2 in Vollmer et al. 2019]
		vector<pair<numm, vector<ind>>> topTemplates;
		numm minGain = 0;
		for (auto iter = patternCandidates.begin(); iter != patternCandidates.end(); ++iter) {
			numm matchInSample = 0;
			vector<ind> allocatedTuple = iter->first;

			if (removeInactivatedPatterns) {
				bool inactivatedPattern = false;
				// std::stringstream stream;
				for (ind d = 0; d < D; ++d) {
					// stream << attributeNames[d] << "=" << allocatedTuple[d] << ", ";
					if ((allocatedTuple[d] == 0) && (allocatedTuple[D + d] == 0)) {
						inactivatedPattern = true;
						break;
					}
				}
				if (inactivatedPattern) {
					// cout << stream.str() << "\n";
					continue;
				}
			}

			for (ind s = 0; s < sample.size(); ++s) {
				if (match(sample[s], allocatedTuple)) { matchInSample += 1; }
			}
			iter->second[2 * O] /= (numm)matchInSample;
			for (ind o = 0; o < O; ++o) {
				iter->second[2 * o] /= (numm)matchInSample;
				iter->second[2 * o + 1] /= (numm)matchInSample;
			}
			numm maxGain = 0;
			for (ind o = 0; o < O; ++o) {
				numm trueRate = iter->second[2 * o] / (numm)iter->second[2 * O];
				numm expectedRate = iter->second[2 * o + 1] / (numm)iter->second[2 * O];
				numm supportRate = (numm)iter->second[2 * O] / (numm)totalN;

				if ((trueRate != 0) && (expectedRate != 0) && (supportRate >= minSupport)) {
					numm gain = iter->second[2 * O] * trueRate *log(trueRate / expectedRate);
					for (ind o2 = 0; o2 < O; ++o2) {
						if (o2 != o) {
							numm tr = iter->second[2 * o2] / (numm)iter->second[2 * O];
							numm oldER = iter->second[2 * o2 + 1] / (numm)iter->second[2 * O];
							numm newER = (oldER / (1.0 - expectedRate)) * (1.0 - trueRate);
							if ((tr != 0) && (oldER != 0) && (newER != 0)) {
								gain += iter->second[2 * O] * tr * (log(tr / oldER) - log(tr / newER));
							}
						}
					}
					maxGain = max(maxGain, gain);
				}
			}
			if (maxGain > minGain) {
				topTemplates.emplace_back(maxGain, allocatedTuple);
				if (topTemplates.size() > 10) {
					sort(topTemplates.rbegin(), topTemplates.rend());
					topTemplates.resize(10);
					minGain = topTemplates[9].first;
				}
			}
		}

		if (topTemplates.size() == 0) {
			cout << "No pattern candidates generated, should end the process!" << "\n";
			return false;
		}

		// insert best pattern into the explanation table
		vector<ind> bestPattern(2 * D + 3, 0);
		numm bestGain = 0;
		evalStraight(topTemplates[0].second, bestPattern, bestGain); // select best outcome for the pattern

		// If the pattern is already added to the table, the process should end: 
		for (ind p = 0; p < patterns.size(); ++p) {
			  vector<ind> pat = patterns[p];
			  if (patternsEqual(pat, bestPattern)) {
				  cout << "New pattern already exists in the table, should end the process!" << "\n";
				  return false;
			  }
		}

		addRule(bestPattern);
		iterativeScaling();
		auto end = chrono::high_resolution_clock::now();
		if (verbose) {
			numm currentKL = KLDivergence();
			cout << "FL Original " << sampleSize << "," << rules + 1 << "," << chrono::duration_cast<chrono::milliseconds>(end - start).count()<<"," << chrono::duration_cast<chrono::milliseconds>(end - globalStart).count() << ",";
			cout << recentKL - currentKL << "," << baseKL - currentKL << "," << currentKL << "\n";
			recentKL = currentKL;
		}
		return true;
	}
}

void Lighthouse::printTable(numm baseKL, string fileName) // print explanation table to console
{
	ofstream file;
	file.open(fileName, ofstream::trunc);

	numm KL = KLDivergence();
	numm gain = baseKL - KL;
	cout << "\n===[Table Size: " << patterns.size() << " | Base KL-Div: " << baseKL << " | KL-Div: " << KL << " | Gain: " << gain << "]===\n";
	for (ind d = 0; d < D; ++d) {
		cout<<attributeNames[d]<<",";
		file << attributeNames[d] << ",";
	}
	cout<<attributeNames[D]<<"\n";
	file << attributeNames[D] << ",support,confidence" << "\n" << flush;

	for (ind p = 0; p < patterns.size(); ++p) {
		for (ind d = 0; d < D; ++d) {
			if ((patterns[p][d] == 0) && (patterns[p][D + d] == keyToAttribute[d].size() - 1) && (p >= O)) {
				cout << "[*],";
				if (p >= O) {   // if not among two first general distribution patterns
					file << "-1,";
				}
			} 
			else {
				numm featureValue = 0;
				numm firstValue = keyToAttribute[d].find(patterns[p][d])->second;
				numm secondValue = keyToAttribute[d].find(patterns[p][D+d])->second;
				if (firstValue == secondValue) {
					featureValue = firstValue;
				}
				else {     // handling the binned features case where more than one feature value may be possible in a pattern
					featureValue = ((secondValue - firstValue) / 2);
				}

				cout << "[" << firstValue << "-" << secondValue << "],";
				if (p >= O) {
					file << featureValue << ",";
				}
			}
		}

		string outcome = keyToOutcome.find(patterns[p][2 * D])->second; 
		ind support = patterns[p][2 * D + 1];
		numm supportPercentage = (numm)support / (numm)totalN;
		numm confidence = (numm)patterns[p][2 * D + 2] / (numm)support;

		cout << outcome << "  ; Sup: " << support << " Prec: " << confidence << "\n";
		if (p >= O) {
			file << outcome << "," << supportPercentage << "," << confidence << "\n" << flush;
		}
	}
}

string Lighthouse::getTable() //returns the entire explanation table as string
{
	string result = "";
	result += "\n===[Table Size: " + to_string(patterns.size()) + " | KL-Div: " + to_string(KLDivergence()) + "]===\n";
	for (ind d = 0; d < D; ++d) {
		result += attributeNames[d] + ",";
	}
	result += attributeNames[D] + "\n";
	for (ind p = 0; p < patterns.size(); ++p) {
		for (ind d = 0; d < D; ++d) {
			if ((patterns[p][d] == 0) && (patterns[p][D + d] == keyToAttribute[d].size() - 1) && (p >= O)) {
				result += + "[*],";
			}
			else {
				result += + "[" + to_string(keyToAttribute[d].find(patterns[p][d])->second) + "-" + to_string(keyToAttribute[d].find(patterns[p][D + d])->second) + "],";
			}
		}
		result += keyToOutcome.find(patterns[p][2 * D])->second + "  ; Sup: " + to_string(patterns[p][2 * D + 1]) + " Prec: " + to_string((numm)patterns[p][2 * D + 2] / (numm)patterns[p][2 * D + 1]) + "\n";
	}
	return result;
}

void Lighthouse::LENS(ind numberRules, bool verbose, ind samplesize, ind FLCandidates, ind mode)
{
	const ind patternCutoff = 6; // define limit for dimensions of pattern expansion, i.e., if a simple pattern from Flashlight has 6 or more constants it is not considered for expansion
	// internal counting gain and time for verbose mode
	auto globalStart = chrono::high_resolution_clock::now();
	numm baseKL = KLDivergence();
	numm recentKL = baseKL;
	for (ind rules = 0; rules < numberRules; ++rules) {
		auto start = chrono::high_resolution_clock::now();

		// draw sample for flashlight
		vector<vector<ind>> sample(samplesize);
		generateSample(sample, samplesize);

		// generate LCA table immediatly with aggregates [cf. Table 7 in El Gebaly et al. 2014]
		unordered_map<vector<ind>, vector<numm>, ind_vector_hasher> patternCandidates; // [AttributesLow, AttributesHigh] -> [Out1Count, Out1Exp, Out2Count, ..., OutxExp, TotalCount]
		for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
			for (ind s = 0; s < sample.size(); ++s) {
				vector<ind> protoPattern(2 * D, 0);
				for (ind k = 0; k < D; ++k) {
					if (iter->first[k] == sample[s][k]) {
						protoPattern[k] = sample[s][k];
						protoPattern[D + k] = sample[s][k];
					}
					else {
						protoPattern[k] = 0;
						protoPattern[D + k] = attributeToKey[k].size() - 1;
					}
				}
				addSimpleAncestors(protoPattern, patternCandidates, 0, iter->second);
			}
		}

		vector<pair<numm, vector<ind>>> topTemplates;
		numm minGain = 0;
		for (auto iter = patternCandidates.begin(); iter != patternCandidates.end(); ++iter) {
			ind patternDim = 0;
			for (ind d = 0; d < D; ++d) {
				if (iter->first[d] == iter->first[D + d]) {++patternDim;}
			}
			if (patternDim < patternCutoff) {
				numm matchInSample = 0;
				vector<ind> allocatedTuple = iter->first;
				for (ind s = 0; s < sample.size(); ++s) {
					if (match(sample[s], allocatedTuple)) { matchInSample += 1; }
				}
				iter->second[2 * O] /= (numm)matchInSample;
				for (ind o = 0; o < O; ++o) {
					iter->second[2 * o] /= (numm)matchInSample;
					iter->second[2 * o + 1] /= (numm)matchInSample;
				}
				numm maxGain = 0;
				for (ind o = 0; o < O; ++o) {
					numm trueRate = iter->second[2 * o] / (numm)iter->second[2 * O];
					numm expectedRate = iter->second[2 * o + 1] / (numm)iter->second[2 * O];
					if ((trueRate != 0) && (expectedRate != 0)) {
						numm gain = iter->second[2 * O] * trueRate *log(trueRate / expectedRate);
						for (ind o2 = 0; o2 < O; ++o2) {
							if (o2 != o) {
								numm tr = iter->second[2 * o2] / (numm)iter->second[2 * O];
								numm oldER = iter->second[2 * o2 + 1] / (numm)iter->second[2 * O];
								numm newER = (oldER / (1.0 - expectedRate)) * (1.0 - trueRate);
								if ((tr != 0) && (oldER != 0) && (newER != 0)) {
									gain += iter->second[2 * O] * tr * (log(tr / oldER) - log(tr / newER));
								}
							}
						}
						maxGain = max(maxGain, gain);
					}
				}
				if (maxGain > minGain) {
					topTemplates.emplace_back(maxGain, allocatedTuple);
					if (topTemplates.size() > FLCandidates) {
						sort(topTemplates.rbegin(), topTemplates.rend());
						topTemplates.resize(FLCandidates);
						minGain = topTemplates[FLCandidates - 1].first;
					}
				}
			}
		}
		vector<ind> bestPattern(2 * D + 3, 0);
		numm bestGain = 0;
		for (auto iter = topTemplates.begin(); iter != topTemplates.end(); ++iter) {
			/////// BuildAdaptedCube -- Note that the cube is build compactely and efficiently, i.e., aggregation over non-constant attributes and the ranges occurs with the first pass of the data.
			// Find Thresholds
			vector<ind> dimIndices; // which dimensions is this cube build with
			vector<vector<ind>> dimThresholds; // thresholds for cumulative sum aggregration per dimenion
			vector<ind> realcenters;  // list of values per attribute that are the center of expansion
			vector<ind> transformedCenters; // as centers but as index of the aggregated buckets of the cumulative cube
			
			for (ind d = 0; d < D; ++d) {
				if (iter->second[d] == iter->second[D + d]) {
					ind center = iter->second[d];
					vector<ind> localThresholds = {center};
					ind smaller = center;
					for (ind steps = 1; smaller > 0; steps *= 2) {
						(smaller > steps) ? (smaller = smaller - steps) : (smaller = 0);
						localThresholds.push_back(smaller);
					}
					ind larger = center;
					for (ind steps = 1; larger < keyToAttribute[d].size() - 1; steps *= 2) {
						// MY STUFF???
						larger = min(larger + steps, (int)keyToAttribute[d].size() - 1);
						// larger = min(larger + steps, keyToAttribute[d].size() - 1);
						localThresholds.push_back(larger);
					}
					sort(localThresholds.begin(), localThresholds.end());
					dimIndices.push_back(d);
					dimThresholds.push_back(localThresholds);

					realcenters.push_back(center);
					transformedCenters.push_back(distance(localThresholds.begin(), find(localThresholds.begin(), localThresholds.end(), center)));
				}
			}
			ind patternDim = dimIndices.size();
			vector<ind> localBestPattern;
			// Build Cube
			// First pass on the data and consider which cell of the cumulative cube the tuples fall into, then accumulate values across dimensions
			// For efficiency, the high-dimensional cube (with variable dimension count) is built as one-dimensional vector using just multiplicative indexing
			ind cubeSize = 1;
			for (ind d = 0; d < patternDim; ++d) {
				cubeSize *= dimThresholds[d].size();
			}
			auto cumulativeCubeTrue = vector<vector<ind>>(O, vector<ind>(cubeSize, 0));
			auto cumulativeCubeModel = vector<vector<numm>>(O, vector<numm>(cubeSize, 0));
			auto cumulativeCubeSupport = vector<ind>(cubeSize, 0);
			for (auto iter = tuples.begin(); iter != tuples.end(); ++iter) {
				ind cubeID = 0;
				ind multiplier = 1;
				for (ind d = 0; d < patternDim; ++d) {
					ind value = iter->first[dimIndices[d]];
					ind bucket = 0;
					if (value < realcenters[d]) {
						while (value >= dimThresholds[d][bucket+1]) {++bucket;}
					} else {
						bucket = transformedCenters[d];
						while (value > dimThresholds[d][bucket]) { ++bucket; }
					}
					cubeID += multiplier*bucket;
					multiplier *= dimThresholds[d].size();
				}
				cumulativeCubeSupport[cubeID] += iter->second[O];
				for (ind o = 0; o < O; ++o) {
					cumulativeCubeModel[o][cubeID] += iter->second[O] * RCT.find(iter->second[O + 1])->second[o];
					cumulativeCubeTrue[o][cubeID] += iter->second[o];
				}
			}
			ind stepwidth = 1;
			for (ind d = 0; d < patternDim; ++d) {
				for (ind i = 0; i + stepwidth < cumulativeCubeSupport.size(); ++i) {
					ind target = i + stepwidth;
					if (target % (stepwidth * dimThresholds[d].size()) == 0) {
						i += (stepwidth - 1);
					} else {
						cumulativeCubeSupport[target] += cumulativeCubeSupport[i];
						for (ind o = 0; o < O; ++o) {
							cumulativeCubeModel[o][target] += cumulativeCubeModel[o][i];
							cumulativeCubeTrue[o][target] += cumulativeCubeTrue[o][i];
						}
					}
				}
				stepwidth *= dimThresholds[d].size();
			}
			// Greedy BFS Exploration of patterns
			// Note: patterns contain only those dimensions that are present in the cube (i.e. constants in the original, simple pattern)
			vector<ind> protoPattern(2*patternDim);
			for (ind d = 0; d < patternDim; ++d) {
				protoPattern[d] = transformedCenters[d];
				protoPattern[patternDim+d] = transformedCenters[d];
			}
			unordered_map<vector<ind>, numm, ind_vector_hasher> expandablePatterns;
			expandablePatterns.emplace(protoPattern, 0);
			while (expandablePatterns.size() > 0) {
				unordered_map<vector<ind>, numm, ind_vector_hasher> nextExpandablePatterns;
				for (auto iter = expandablePatterns.begin(); iter != expandablePatterns.end(); ++iter) {
					numm currGain = 0;
					vector<ind> currPattern = iter->first;
					//evaluate gain with cumulative cubes
					ind support = 0, supportN = 0;
					vector<ind> trueCount(O, 0), trueCountN(O, 0);
					vector<numm> modelCount(O, 0), modelCountN(O, 0);
					for (ind i = 0; i < pow(2, patternDim); ++i) {
						ind cubeID = 0;
						ind multiplier = 1;
						ind index = 0;
						bool valid = true;
						for (ind d = 0; d < patternDim; ++d) {
							if ((i >> d & 1) && (currPattern[d] == 0)) { valid = false; }
							else { ((i >> d & 1) ? (index += multiplier*(currPattern[d] - 1)) : (index += multiplier * currPattern[patternDim + d])); }
							multiplier *= dimThresholds[d].size();
						}
						if (valid) {
							// MY STUFF
							if ((_popcnt64(i) % 2) == 0) {   // __popcnt in Windows, __popcnt64 in 64-bit VC++
								support += cumulativeCubeSupport[index];
								for (ind o = 0; o < O; ++o) {
									trueCount[o] += cumulativeCubeTrue[o][index];
									modelCount[o] += cumulativeCubeModel[o][index];
								}
							}
							else {
								supportN += cumulativeCubeSupport[index];
								for (ind o = 0; o < O; ++o) {
									trueCountN[o] += cumulativeCubeTrue[o][index];
									modelCountN[o] += cumulativeCubeModel[o][index];
								}
							}
						}
					}
					numm supportF = (support - supportN);
					for (ind o = 0; o < O; ++o) {
						numm trueRate = (numm)(trueCount[o] - trueCountN[o]) / supportF;
						numm expRate = (numm)(modelCount[o] - modelCountN[o]) / supportF;
						if ((trueRate != 0) && (expRate != 0)) {
							numm gain = supportF * trueRate *log(trueRate / expRate);
							for (ind o2 = 0; o2 < O; ++o2) {
								if (o2 != o) {
									numm tr = (numm)(trueCount[o2] - trueCountN[o2]) / supportF;
									numm oldER = (numm)(modelCount[o2] - modelCountN[o2]) / supportF;
									numm newER = (oldER / (1.0 - expRate)) * (1.0 - trueRate);
									if ((tr != 0) && (oldER != 0) && (newER != 0)) {
										gain += supportF * tr * (log(tr / oldER) - log(tr / newER));
									}
								}
							}
							currGain = max(currGain, gain);
							if (gain > bestGain) {
								bestGain = gain;
								localBestPattern = currPattern;
								localBestPattern.push_back(o);
								localBestPattern.push_back(support - supportN);
								localBestPattern.push_back(trueCount[o] - trueCountN[o]);
							}
						}
					}
					if (currGain > iter->second) {
						for (ind d = 0; d < patternDim; ++d) {
							if (currPattern[d] > 0) {
								vector<ind> nextPattern = currPattern;
								nextPattern[d] -= 1;
								auto ref = nextExpandablePatterns.emplace(nextPattern, currGain);
							}
							if (currPattern[patternDim+d] < dimThresholds[d].size()-1) {
								vector<ind> nextPattern = currPattern;
								nextPattern[patternDim +d] += 1;
								auto ref = nextExpandablePatterns.emplace(nextPattern, currGain);
							}
						}
					}
				}
				expandablePatterns = nextExpandablePatterns;
			}
			// construct best pattern from the small pattern, i.e. fill range of remaining attributes with wildcards
			if (localBestPattern.size() > 0) {
				for (ind d = 0; d < D; ++d) {
					bestPattern[d] = 0;
					bestPattern[D + d] = keyToAttribute[d].size() - 1;
				}
				for (ind d = 0; d < patternDim; ++d) {
					bestPattern[dimIndices[d]] = dimThresholds[d][localBestPattern[d]];
					bestPattern[D + dimIndices[d]] = dimThresholds[d][localBestPattern[patternDim + d]];
				}
				for (ind extras = 0; extras < 3; ++extras) { bestPattern[2*D+extras] =  localBestPattern[2*patternDim+extras]; }
			}
		}
		// insert best pattern into the explanation table
		addRule(bestPattern);
		iterativeScaling();
		auto end = chrono::high_resolution_clock::now();
		if (verbose) {
			numm currentKL = KLDivergence();
			cout << "FL+OptGrow " << samplesize << ";" <<FLCandidates<<"," << rules + 1 << "," << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "," << chrono::duration_cast<chrono::milliseconds>(end - globalStart).count() << ",";
			cout << recentKL - currentKL << "," << baseKL - currentKL << "," << currentKL << "\n";
			recentKL = currentKL;
		}
	}
}
