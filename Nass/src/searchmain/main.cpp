#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <cstdlib>
#include <string>

#include "nassged.h"
#include "inves.h"
#include "datafile.h"
#include "dataset.h"
#include "nass.h"

using namespace std;

void mkdirp_search(const string& dir)
{
#ifdef _WIN32
	mkdir(dir.c_str());
#else
	mkdir(dir.c_str(), 0755);
#endif
}

string extractDatasetName(const char* data_file)
{
	string path(data_file);
	size_t last_sep = path.find_last_of("/\\");
	string filename = (last_sep != string::npos) ? path.substr(last_sep + 1) : path;

	size_t dot = filename.find_last_of('.');
	string basename = (dot != string::npos) ? filename.substr(0, dot) : filename;

	if(basename == "db" || basename == "DB" || basename == "data" || basename == "database"){
		if(last_sep != string::npos){
			string parent = path.substr(0, last_sep);
			size_t parent_sep = parent.find_last_of("/\\");
			basename = (parent_sep != string::npos) ? parent.substr(parent_sep + 1) : parent;
		}
	}
	return basename;
}

int main(int argc, char* argv[])
{
	if(argc < 3){
		cerr << "usage: nass threshold data_file [index_file] [-q queries_file]" << endl;
		exit(1);
	}

	// parse the specified threshold
	nass::threshold = 0;
	for(unsigned i = 0; argv[1][i] != '\0'; i++)
		if(argv[1][i] >= '0' && argv[1][i] <= '9')
			nass::threshold = nass::threshold*10 + (argv[1][i] - '0');
		else{
			cerr << "usage: nass threshold data_file [index_file] [-q queries_file]" << endl;
			exit(1);
		}

	// parse optional arguments
	const char* indexfile = NULL;
	char* queries_file = NULL;
	for(int i = 3; i < argc; i++){
		if(strcmp(argv[i], "-q") == 0){
			if(i + 1 < argc) queries_file = argv[++i];
			else{ cerr << "-q requires a filename argument" << endl; exit(1); }
		}
		else if(indexfile == NULL)
			indexfile = argv[i];
		else{
			cerr << "usage: nass threshold data_file [index_file] [-q queries_file]" << endl;
			exit(1);
		}
	}

	DataFile* file = new SyntheticFile(argv[2]);
	DataSet* dataset = DataSet::getInstance();
	dataset->buildDataSet(*file);
	file->close();
	delete file;

	Workload workload;
	if(queries_file != NULL)
		workload.readWorkload(queries_file);
	else
		workload.generateWorkload();

	nass searcher(indexfile);

	// generate log path: logs/{dataset}/search_tau{threshold}.log
	string ds_name = extractDatasetName(argv[2]);
	string log_dir = "logs";
	mkdirp_search(log_dir);
	log_dir += "/" + ds_name;
	mkdirp_search(log_dir);
	string log_file = log_dir + "/" + ds_name + "_search_tau" + to_string(nass::threshold) + ".log";

	SearchMeta meta;
	meta.data_file = argv[2];
	meta.queries_file = queries_file ? queries_file : "(random 100)";
	meta.index_file = indexfile ? indexfile : "";
	meta.log_file = log_file;

	searcher.run(workload, meta);
	cout << endl << flush;

	// all graph (wrapper) objects should be destroyed before calling finishUp()
	DataSet::finishUp();

	return 0;
}
