#include "Includes.h"
#include "StaticFunctions.h"
#include "Lighthouse.h"

int main(int argc, char *argv[])
{
    if (argc != 8) {
		cout << "Not enough arguments: " << argc << "\n";
        exit(-1);
	}
    
    string datasetName = argv[1];
    string datasetFile = argv[2];
    int numFeatures = atoi(argv[3]);
	int numPatterns = atoi(argv[4]);
	bool removeInactivatedPatterns = (atoi(argv[5]) == 1) ? true : false;
	string patternsFile = argv[6];
	numm minSupportParam = atof(argv[7]);
 
    //cout << datasetName << " - " << datasetFile << " - " << numFeatures << " - " << numPatterns << " - " << patternsFile << " - " << minSupportParam << "\n"; 

	vector<tuple<string, string, vector<ind>, ind>> DataSets; //Name, Filepath, inputcolumns, outputcolumn
	vector<ind> attributeIndices;
    for (int i = 0; i < numFeatures; i++) {
        attributeIndices.push_back(i);
    }
 
    DataSets.emplace_back(datasetName, datasetFile, attributeIndices, numFeatures);

	const ind maxRules = numPatterns; // maximum explanation table size
	vector<ind> tablesizes = {maxRules}; // explanation table sizes at which to evaluate time and gain
	vector<numm> minSupportParams = {minSupportParam}; // {0.01, 0.03, 0.05, 0.1, 0.15, 0.2};

	const bool internalPrint = true; // some methods offer internal prints
	const bool externalPrint = true;  // log results as blackbox evaluation
	const bool printTables = true;	  // print final explanation table to console

	const bool TOGGLE_FL = true;   // run Flashlight on Data
	const bool TOGGLE_LENS = false; // run LENS on Data

	string logfile = "./logs/log.csv"; // file to save gain/time results
	ofstream ofile;
	ofile.open(logfile, ofstream::app);
	
	ofile << "Data, Method, TableSize, Time Total, Gain Total, Remaining Divergence,Parameters...\n" << flush;
	for (ind repetitions = 0; repetitions < 1; ++repetitions) 
	{ 
		// enable experiment repetition
		for (ind i = 0; i < DataSets.size(); ++i)
		{ 
			// itrerate datasets and parse information
			string dataName = get<0>(DataSets[i]);
			//cout << "=======" << dataName << "========\n";
			string file = get<1>(DataSets[i]);
			vector<ind> dataColumns = get<2>(DataSets[i]);
			ind outcome = get<3>(DataSets[i]);

			// Obtain Reference KL-Divergence
			Lighthouse ref(file, dataColumns, outcome);
			numm baseKL = ref.KLDivergence();

			//== ClassicFL == Parameters: Samplesize//
			if (TOGGLE_FL)
			{
				vector<ind> configsFL = {16}; // Enable easy way to test multiple settings for sample size
				for (ind s : configsFL)
				{
					for (numm minSupport : minSupportParams) {
						
						Lighthouse A(file, dataColumns, outcome);
						cout << "== FL Classic -- " << s << " -- " << minSupport << " ==\n";
						auto start = chrono::high_resolution_clock::now();
						for (ind r = 1; r <= maxRules; ++r)
						{
							bool shouldContinue = A.originalFlashlight(1, s, minSupport, removeInactivatedPatterns, internalPrint);
							if (!shouldContinue) {
								break;
							}
							if (externalPrint && (find(tablesizes.begin(), tablesizes.end(), r) != tablesizes.end()))
							{ 
								// test whether current table size should be evaluated
								auto now = chrono::high_resolution_clock::now();
								numm KL = A.KLDivergence();
								ofile << dataName << ",Flashlight," << r << "," << chrono::duration_cast<chrono::milliseconds>(now - start).count() << "," << baseKL - KL << "," << KL << ",";
								ofile << s << "\n"
									<< flush;

								//cout << "Patterns: " << r << ", Base KL: " << baseKL << ", KL: " << KL << ", Gain: " << baseKL - KL << endl;
							}
						}
						if (printTables)
						{
							int precision = 2;
							if (std::fmod(minSupport, 0.1) == 0)
	   							precision = 1;
							std::stringstream stream;
							stream << std::fixed << std::setprecision(precision) << minSupport;
							std::string minSupportStr = stream.str();
							//string patternsFile = "exp_patterns_" + minSupportStr + ".csv";
							A.printTable(baseKL, patternsFile);
						}
					}
				}
			}

			//== BottomUp with Cube == Parameters: Samplesize, Candidates//
			if (TOGGLE_LENS)
			{
				vector<pair<ind, ind>> configsLENS; // Easy way to test multiple parameter settings for LENS
				configsLENS.emplace_back(4, 16);
				configsLENS.emplace_back(16, 32);
				for (auto config : configsLENS)
				{
					ind s = config.first;
					ind k = config.second;
					Lighthouse A(file, dataColumns, outcome);
					cout << "== Bottom Up Cube -- " << s << ";" << k << " ==\n";
					auto start = chrono::high_resolution_clock::now();
					for (ind r = 1; r <= maxRules; ++r)
					{
						A.LENS(1, internalPrint, s, k, 0);
						if (externalPrint && (find(tablesizes.begin(), tablesizes.end(), r) != tablesizes.end()))
						{
							auto now = chrono::high_resolution_clock::now();
							numm KL = A.KLDivergence();
							ofile << dataName << ",LENS," << r << "," << chrono::duration_cast<chrono::milliseconds>(now - start).count() << "," << baseKL - KL << "," << KL << ",";
							ofile << s << "," << k << "\n"
								  << flush;
						}
					}
					if (printTables)
					{
						A.printTable(baseKL, patternsFile);
					}
				}
			}
		}
		ofile.close();
	}
}
