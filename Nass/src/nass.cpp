#include <sstream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <atomic>
#include <thread>

#include "nass.h"
#include "nassged.h"

//statistics
std::atomic<long long> cands_final;
std::atomic<long long> push_count;

int nass::threshold = 0;

nass::nass(const char* filename, bool load_index)
{
	this->indexfile = filename;

	this->use_index = false;
	this->nassIndex = NULL;
	this->sampled = NULL;
	this->index_max_threshold = 0;

	if(load_index && indexfile && loadNassIndex())
		use_index = true;

	clearStats();
}

nass::~nass()
{
	delete [] nassIndex;
	delete [] sampled;
}


void nass::clearStats()
{
	cands_final = 0;
	push_count = 0;
	res_vec.clear();
}

void nass::run(Workload& workload)
{
	SearchMeta meta;
	run(workload, meta);
}

void nass::run(Workload& workload, const SearchMeta& meta)
{
	clearStats();

	struct QueryStat {
		long long candidates;
		long long mappings;
		long long results;
		long long time_us;
	};

	unsigned nq = workload.size();
	vector<QueryStat> qstats(nq);
	vector<vector<int> > thread_results(NTHREADS);

	NassGED::initialize();

	std::atomic<unsigned> next_query(0);
	std::atomic<unsigned> done_count(0);
	TIMESTAMP wall_start, wall_end;
	GETTIME(&wall_start);

	auto worker = [&](int wid) {
		while(true){
			unsigned qi = next_query.fetch_add(1);
			if(qi >= nq) break;

			TIMESTAMP q_start, q_end;
			GETTIME(&q_start);

			long long prev_cands = cands_final.load();
			long long prev_push = push_count.load();
			unsigned prev_res = thread_results[wid].size();

			search(workload[qi], wid, thread_results[wid]);

			GETTIME(&q_end);
			long long q_time = (q_end.tv_sec - q_start.tv_sec) * 1000000LL
			                  + (q_end.tv_usec - q_start.tv_usec);

			qstats[qi].candidates = cands_final.load() - prev_cands;
			qstats[qi].mappings = push_count.load() - prev_push;
			qstats[qi].results = thread_results[wid].size() - prev_res;
			qstats[qi].time_us = q_time;

			unsigned d = done_count.fetch_add(1) + 1;
			unsigned pct = d * 100 / nq;
			if(d == nq || pct > (d - 1) * 100 / nq){
				fprintf(stderr, "\rNass: %u%%", pct);
				fflush(stderr);
			}
		}
	};

	std::thread* workers[NTHREADS];
	for(int i = 0; i < NTHREADS; i++)
		workers[i] = new std::thread(worker, i);
	for(int i = 0; i < NTHREADS; i++){
		workers[i]->join();
		delete workers[i];
	}

	GETTIME(&wall_end);
	long long elapsed = (wall_end.tv_sec - wall_start.tv_sec) * 1000000LL
	                   + (wall_end.tv_usec - wall_start.tv_usec);

	NassGED::finalize();

	// merge results
	for(int i = 0; i < NTHREADS; i++)
		res_vec.insert(res_vec.end(), thread_results[i].begin(), thread_results[i].end());

	// console summary
	fprintf(stderr, "\r                    \r");
	cout << "===================== Results =====================" << endl;
	cout << "GED threshold: " << threshold << endl;
	cout << "# queries: " << nq << endl;
	cout << "# threads: " << NTHREADS << endl;
	cout << "# total candidates: " << cands_final.load() << endl;
	cout << "# total mappings: " << push_count.load() << endl;
	cout << "# total results: " << res_vec.size() << endl;
	cout << "Wall time: ";
	fprintf(stdout, "%lld.%03lld seconds\n", elapsed/1000000, (elapsed % 1000000)/1000);
	cout << "===================================================" << endl;

	// write log file
	if(!meta.log_file.empty()){
		ofstream log(meta.log_file);
		if(log.is_open()){
			time_t now = time(NULL);
			struct tm* t = localtime(&now);
			char timebuf[64];
			strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", t);

			DataSet* dataset = DataSet::getInstance();

			log << "=== Nass Search Log ===" << endl;
			log << "Date: " << timebuf << endl;
			log << "Dataset: " << meta.data_file << endl;
			log << "Queries: " << meta.queries_file << endl;
			log << "Index: " << (meta.index_file.empty() ? "(none)" : meta.index_file) << endl;
			log << "Threshold (tau): " << threshold << endl;
			log << "Num threads: " << NTHREADS << endl;
			log << "Num graphs in DB: " << dataset->numGraphs() << endl;
			log << "Num queries: " << nq << endl;
			log << endl;

			log << "=== Per-Query Results ===" << endl;
			log << left << setw(8) << "Query"
			    << right << setw(12) << "Candidates"
			    << setw(12) << "Mappings"
			    << setw(10) << "Results"
			    << setw(16) << "Time(ms)" << endl;
			log << string(58, '-') << endl;

			for(unsigned i = 0; i < nq; i++){
				log << left << setw(8) << i
				    << right << setw(12) << qstats[i].candidates
				    << setw(12) << qstats[i].mappings
				    << setw(10) << qstats[i].results
				    << setw(13) << qstats[i].time_us / 1000
				    << "." << setfill('0') << setw(3) << qstats[i].time_us % 1000
				    << setfill(' ') << endl;
			}

			log << endl;
			log << "=== Summary ===" << endl;
			log << "Total candidates: " << cands_final.load() << endl;
			log << "Total mappings: " << push_count.load() << endl;
			log << "Total results: " << res_vec.size() << endl;
			log << "Total wall time: " << elapsed/1000000 << "."
			    << setfill('0') << setw(3) << (elapsed % 1000000)/1000
			    << setfill(' ') << " seconds" << endl;
			if(nq > 0){
				long long sum_query_time = 0;
				for(unsigned i = 0; i < nq; i++)
					sum_query_time += qstats[i].time_us;
				long long avg = sum_query_time / nq;
				log << "Avg time per query: " << avg/1000000 << "."
				    << setfill('0') << setw(3) << (avg % 1000000)/1000
				    << setfill(' ') << " seconds" << endl;
			}
			log << "========================" << endl;
			log.close();

			cout << "Log written to: " << meta.log_file << endl;
		}
	}
}

bool gtid(const pair<int, int>& c1, const pair<int, int>& c2)
{
	return c1.first < c2.first;
}

bool gtlb(const pair<int, int>& c1, const pair<int, int>& c2)
{
	return c1.second < c2.second;
}

unsigned nass::regenCandidates(vector<pair<int, int> >& candidates, unsigned pos, int res, int distance, vector<int>& results)
{
	int max_threshold = index_max_threshold;
	if(!sampled[res]) max_threshold -= 1;

	results.push_back(res);
	if(distance + threshold > max_threshold) return pos;

	std::sort(candidates.begin() + pos, candidates.end(), gtid);

	vector<pair<int,int> > tmp;
	unsigned ptr1 = pos, ptr2 = 0;

	while(ptr1 < candidates.size() && ptr2 < nassIndex[res].size()){
		int gdist = nassIndex[res][ptr2].second;
		bool inexact = (gdist < 0);
		if(gdist == INEXACT_ZERO) gdist = 0;
		if(inexact) gdist = gdist * -1;

		if(gdist > threshold + distance) ptr2++;
		else if(nassIndex[res][ptr2].first < candidates[ptr1].first) ptr2++;
		else if(nassIndex[res][ptr2].first > candidates[ptr1].first) ptr1++;
		else{
			if(gdist + distance <= threshold && !inexact)
				results.push_back(nassIndex[res][ptr2].first);
			else tmp.push_back(candidates[ptr1]);

			ptr1++; ptr2++;
		}
	}
	std::swap(candidates, tmp);

	std::sort(candidates.begin(), candidates.end(), gtlb);

	return 0;
}

void nass::basicFilter(coregraph* x, vector<pair<int, int> >& candidates, int threshold, int begin)
{
	DataSet* dataset = DataSet::getInstance();

	for(unsigned i = begin; i < dataset->numGraphs(); i++){
		coregraph* y = dataset->graphAt(i);
		if(x == y) continue; // skip the same graph
		if(x->sizeFilter(y) > unsigned(threshold)) continue;

		int lb = x->labelFilter(y); //gx.unmappedErrors(gy);
		if(lb <= threshold)
			candidates.push_back(pair<int, int>(i, lb));
	}
}

#include <unistd.h>
void nass::search(coregraph* x, int wid, vector<int>& results)
{
	DataSet* dataset = DataSet::getInstance();

	vector<pair<int, int> > candidates;
	basicFilter(x, candidates, threshold);

	if(use_index) std::sort(candidates.begin(), candidates.end(), gtlb);

	unsigned i = 0;
	while(i < candidates.size()){
		coregraph* y = dataset->graphAt(candidates[i].first);

		NassGED* ged = new NassGED(x, y, threshold, NULL, wid);
		int distance = threshold + 1;
		distance = ged->computeGED();
		delete ged;

		if(distance <= threshold){
			if(use_index) i = regenCandidates(candidates, i+1, candidates[i].first, distance, results);
			else results.push_back(candidates[i++].first);
		}
		else i++;
	}
}
