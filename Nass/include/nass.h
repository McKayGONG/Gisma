#ifndef __NASS_H
#define __NASS_H

#include <algorithm>
#include <cstdlib>
#include <list>
#include <mutex>
#include <string>

#include "dataset.h"
#include "coregraph.h"
#include "looptimer.h"
#include "workload.h"

struct SearchMeta {
	string data_file;
	string queries_file;
	string index_file;
	string log_file;
};

struct IndexBuildMeta {
	string data_file;
	string index_file;
	string log_file;
	double timeout_hours;  // 0 = no limit
	IndexBuildMeta() : timeout_hours(24) {}
};

#define PARTIME 1 // incremental partitioning time
#define GEDTIME 2 // ged computation time

extern int NTHREADS;
extern unsigned MEMLIMIT;

class nass
{
public:
	static int threshold;

private:
	loop_timer* timer;
	vector<int> res_vec;

	void clearStats();

private:
	void basicFilter(coregraph* x, vector<pair<int, int> >& candidates, int threshold, int begin = 0);
	unsigned regenCandidates(vector<pair<int, int> >& candidates, unsigned pos, int res, int distance, vector<int>& results);
	void search(coregraph* x, int wid, vector<int>& results);

public:
	nass(const char* filename, bool load_index = true);
	~nass(); //{ delete [] nassIndex; }

public:
	void run(Workload& workload);
	void run(Workload& workload, const SearchMeta& meta);

/******************************************************/
/*                    INDEXING                        */
/******************************************************/
private:
	const char* indexfile;
	int index_max_threshold;
	vector< pair<int, int> >* nassIndex;

	bool use_index;
	bool* sampled;
	bool use_sample;

public:
	void buildNassIndex(bool distributed, char* address, int sampling_rate);
	void buildNassIndex(bool distributed, char* address, int sampling_rate, const IndexBuildMeta& meta);
	bool loadNassIndex(bool verbose = true);

// for measuring indexing time
private:
    TIMESTAMP ts1, ts2;

// for multi-threaded index building
private:
	mutex id_mutex;
	mutex out_mutex;

	int  num_index_worker;
	bool done_indexing;
	bool build_timeout;
	long long timeout_sec;
	unsigned num_indexed;

	unsigned next_graph_id;
	int next_worker_id;

	int getNextGraphIDLocal();
	int getNextGraphID(vector<pair<int, int> >* res = NULL, int id = -1);

	int getWorkerID();
	void outputResult(vector<pair<int, int> >& res, int gid);
	void searchWorker();

	void monitor(); // status mointoring thread

// for distributed index building
private:
	char* server_address;
	int sv_sock; // server socket
	bool distributed; // distributed index building if true
	bool remote; // client if remote == true

#ifndef _WIN32
	int connectToCoordinator(const char* address = NULL);
	int initCoordinator(struct sockaddr_in&, struct sockaddr_in&);

	int getNextGraphIDRemote(vector<pair<int, int> >* res, int id);

	void coordinateLoop();
#endif
};

bool gtid(const pair<int, int>& c1, const pair<int, int>& c2);
bool gtlb(const pair<int, int>& c1, const pair<int, int>& c2);

#endif //__NASS_H
