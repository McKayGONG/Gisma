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

void exit_with_usage()
{
	cerr << "usage: nass-index threshold data_file [index_file] [options]" << endl;
	cerr << "  index_file is optional; auto-generated as {dataset}_tau{threshold}.idx if omitted" << endl;
	cerr << "  -p <threads>   number of threads" << endl;
	cerr << "  -M <MB>        memory limit in MB (default: 2000)" << endl;
	cerr << "  -T <hours>     time limit in hours (default: 24, 0=no limit)" << endl;
	cerr << "usage[client]: nass-index data_file server_addr" << endl;
	exit(1);
}

void mkdirp(const string& dir)
{
#ifdef _WIN32
	mkdir(dir.c_str());
#else
	mkdir(dir.c_str(), 0755);
#endif
}

string extractDatasetNameIdx(const char* data_file)
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

string generateIndexFilename(const char* data_file, int threshold)
{
	string path(data_file);
	size_t last_sep = path.find_last_of("/\\");
	string filename = (last_sep != string::npos) ? path.substr(last_sep + 1) : path;

	size_t dot = filename.find_last_of('.');
	string basename = (dot != string::npos) ? filename.substr(0, dot) : filename;

	// if basename is generic (e.g. "db"), use parent directory name instead
	if(basename == "db" || basename == "DB" || basename == "data" || basename == "database"){
		if(last_sep != string::npos){
			string parent = path.substr(0, last_sep);
			size_t parent_sep = parent.find_last_of("/\\");
			basename = (parent_sep != string::npos) ? parent.substr(parent_sep + 1) : parent;
		}
	}

	// create indexes/{dataset}/ directory structure
	string dir = "indexes";
	mkdirp(dir);
	dir += "/" + basename;
	mkdirp(dir);

	return dir + "/" + basename + "_tau" + to_string(threshold) + ".idx";
}

int main(int argc, char* argv[])
{
	if(argc < 3) exit_with_usage();

	bool distributed = false;
	for(unsigned i = 0; argv[1][i] != '\0'; i++)
		if(argv[1][i] >= '0' && argv[1][i] <= '9') continue;
		else{ distributed = true; break; }

	if(distributed && argc != 3) exit_with_usage();

	char* server_address = NULL;
	char* data_file = NULL;
	char* index_file = NULL;
	string auto_index_file;

	if(distributed){ data_file = argv[1]; server_address = argv[2]; }
	else{
		nass::threshold = atoi(argv[1]);
		data_file = argv[2];

		// argv[3] is index_file if it exists and doesn't start with '-'
		int opt_start = 3;
		if(argc > 3 && argv[3][0] != '-'){
			index_file = argv[3];
			opt_start = 4;
		} else {
			auto_index_file = generateIndexFilename(data_file, nass::threshold);
			index_file = (char*)auto_index_file.c_str();
		}

		int sampling_rate = 100;
		double timeout_hours = 24;
		for(int i = opt_start; i < argc; i += 2){
			if(strcmp(argv[i], "-M") == 0)
				MEMLIMIT = atoi(argv[i+1]);
			else if(strcmp(argv[i], "-p") == 0)
				NTHREADS = atoi(argv[i+1]);
			else if(strcmp(argv[i], "-s") == 0)
				sampling_rate = atoi(argv[i+1]);
			else if(strcmp(argv[i], "-T") == 0)
				timeout_hours = atof(argv[i+1]);
			else if(strcmp(argv[i], "--coordinator") == 0){
				distributed = true;
				i -= 1;
			}
			else{
				cerr << "Unknown option: " << argv[i] << endl;
				exit_with_usage();
			}
		}

		cout << "Index file: " << index_file << endl;

		DataFile* file = new SyntheticFile(data_file);
		DataSet* dataset = DataSet::getInstance();
		dataset->buildDataSet(*file);
		file->close();
		delete file;

		// generate log path
		string ds_name = extractDatasetNameIdx(data_file);
		string log_dir = "logs";
		mkdirp(log_dir);
		log_dir += "/" + ds_name;
		mkdirp(log_dir);
		string log_file = log_dir + "/" + ds_name + "_index_tau" + to_string(nass::threshold) + ".log";

		IndexBuildMeta meta;
		meta.data_file = data_file;
		meta.index_file = index_file;
		meta.log_file = log_file;
		meta.timeout_hours = timeout_hours;

		nass builder(index_file, false);
		builder.buildNassIndex(distributed, server_address, sampling_rate, meta);

		DataSet::finishUp();
		return 0;
	}

	// distributed client path
	DataFile* file = new SyntheticFile(data_file);
	DataSet* dataset = DataSet::getInstance();
	dataset->buildDataSet(*file);
	file->close();
	delete file;

	nass builder(index_file, false);
	builder.buildNassIndex(distributed, server_address, 100);

	DataSet::finishUp();
	return 0;
}
