#include "Application.h"
#include "Graph.h"
#include "Utility.h"
#include "Timer.h"
#include "popl.hpp"

#include <thread>
#include <atomic>
#include <mutex>
#include <set>
#include <fstream>
#include <sstream>

using namespace std;
using namespace popl;

// 读取 Gisma ground truth 文件: 每行 "query_id distance [id, id, ...]".
// 构建 query_id -> {真实 GED <= tau 的 db id 集合}.
static void load_ground_truth_within(const char *file_name, double tau,
                                     map<int, set<int> > &gt) {
	ifstream in(file_name);
	if(!in.is_open()) {
		printf("!!! Could not open ground truth file: %s\n", file_name);
		return;
	}
	string line;
	while(getline(in, line)) {
		istringstream iss(line);
		int qid; double dist; char bracket;
		if(!(iss >> qid >> dist >> bracket) || bracket != '[') continue;
		if(dist > tau) continue;            // 只收 <= tau 的
		int gid;
		while(iss >> gid) {
			gt[qid].insert(gid);
			if(iss.peek() == ',') iss.ignore();
		}
	}
}

void print_usage() {
	printf("Usage: ./ged -h -d database_file -q query_file -m running_mode -p search_paradigm -l lower_bound_method -t ged_threshold\n");
	printf("**** Note that for GED verification, if the returned value is not -1, then it is only an upper bound of (and may be larger than) the exact GED\n\n");
}

ui label2int(const char *str, map<string,ui> &M) {
	if(M.find(string(str)) == M.end()) M[string(str)] = M.size();
	return M[string(str)];
}

ui load_db(const char *file_name, vector<Graph*> &graphs, map<string,ui> &vM, map<string,ui> &eM) {
	FILE *fin = Utility::open_file(file_name, "r");

	const ui MAX_LINE = 1024;
	char line[MAX_LINE];
	if(fgets(line, MAX_LINE, fin) == NULL) {
		fclose(fin);
		return 0;
	}

	// 支持两种图头格式:
	//   原版:  "t C <id>"   (两个字段)
	//   Gisma: "ID <id>"    (一个字段, 顶点/边 label 为整数字符串)
	#define IS_GRAPH_HEADER(L) ((L)[0] == 't' || ((L)[0] == 'I' && (L)[1] == 'D'))

	ui max_n = 0;
	while(IS_GRAPH_HEADER(line)) {
		char buf[128], buf1[128];
		string id;
		if(line[0] == 't') {
			sscanf(line+2, "%s%s", buf1, buf);
			id = string(buf);
		} else { // "ID <id>"
			sscanf(line+2, "%s", buf);
			id = string(buf);
		}
		line[0] = 'x';

		vector<pair<int,ui> > vertices;
		vector<pair<pair<int,int>,ui> > edges;
		while(fgets(line, MAX_LINE, fin) != NULL&&!IS_GRAPH_HEADER(line)) {
			if(line[0] == 'v') {
				int a;
				sscanf(line+2, "%d%s", &a, buf);
				//buf[0] = '1';
				vertices.pb(mp(a, label2int(buf, vM)));
			}
			else if(line[0] == 'e') {
				int a, b;
				sscanf(line+2, "%d%d%s", &a, &b, buf);
				edges.pb(mp(mp(a,b), label2int(buf, eM)));
				edges.pb(mp(mp(b,a), label2int(buf, eM)));
			}
			else printf("!!! Unrecongnized first letter in a line when loading DB!\n");
			line[0] = 'x';
		}

		sort(vertices.begin(), vertices.end());
		for(ui i = 0;i < vertices.size();i ++) assert(vertices[i].first == i);
		if(vertices.size() > max_n) max_n = vertices.size();

		sort(edges.begin(), edges.end());
		for(ui i = 0;i < edges.size();i ++) {
			assert(edges[i].first.first >= 0&&edges[i].first.first < vertices.size());
			assert(edges[i].first.second >= 0&&edges[i].first.second < vertices.size());
			if(i > 0) assert(edges[i].first != edges[i-1].first);
			assert(edges[i].second < eM.size());
		}

		graphs.pb(new Graph(id, vertices, edges));
	}

	fclose(fin);
	return max_n;
}

void generate_queries(const vector<Graph *> &db, vector<ui> &queries, ui q_n) {
	assert(!db.empty());
	srand(time(NULL));
	for(ui i = 0;i < q_n;i ++) queries.pb(rand()%db.size());
}

void write_queries(const char *file_name, const vector<Graph *> &db, const vector<ui> &queries, const map<string,ui> &vM, const map<string,ui> &eM, bool bss) {
	vector<string> vlabels(vM.size());
	vector<string> elabels(eM.size());

	for(pair<string,ui> p: vM) vlabels[p.second] = p.first;
	for(pair<string,ui> p: eM) elabels[p.second] = p.first;

	FILE *fout = Utility::open_file(file_name, "w");

	for(ui i = 0;i < queries.size();i ++) db[queries[i]]->write_graph(fout, vlabels, elabels, bss);

	fclose(fout);
}

int main(int argc, char *argv[]) {

#ifndef NDEBUG
	printf("**** GED (Debug) build at %s %s ***\n", __TIME__, __DATE__);
#else
	printf("**** GED (Release) build at %s %s ***\n", __TIME__, __DATE__);
#endif

	print_usage();

	string mode, paradigm, lower_bound, method;
	int threshold = -1;
	bool print_ged = false;
	int num_workers = 1;
	int q_start = -1, q_end = -1;

	OptionParser op("Allowed options");
	auto help_option = op.add<Switch>("h", "help", "\'produce help message\'");
	auto database_option = op.add<Value<string>>("d", "database", "\'database file name\'");
	auto query_option = op.add<Value<string>>("q", "query", "\'query file name\'");
	auto mode_option = op.add<Value<string>>("m", "mode", "\'running mode\' (search | pair)", "search", &mode);
	auto paradigm_option = op.add<Value<string>>("p", "paradigm", "\'search paradigm\' (astar | dfs)", "astar", &paradigm);
	auto lower_bound_option = op.add<Value<string>>("l", "lower_bound", "\'lower bound method\' (LSa | BMao | BMa)", "BMao", &lower_bound);
	auto threshold_option = op.add<Value<int>>("t", "threshold", "\'threshold for GED verification; if not provided, then GED computation", -1, &threshold);
	op.add<Switch>("g", "ged", "\'print_ged\'", &print_ged);
	op.add<Value<int>>("w", "workers", "\'number of parallel worker threads in search mode (default 1 = serial; 0 = auto-detect cores)\'", 1, &num_workers);
	op.add<Value<int>>("", "q_start", "\'first query index to process in search mode (default -1 = all)\'", -1, &q_start);
	op.add<Value<int>>("", "q_end", "\'last query index (inclusive) to process in search mode (default -1 = all)\'", -1, &q_end);
	auto gt_option = op.add<Value<string>>("", "ground_truth", "\'ground truth file (Gisma format); enables recall/precision reporting in search mode\'");
	long long app_max_iter = 0;
	op.add<Value<long long>>("", "app_max_iter", "\'max A* iterations per pair for App-BMao (default 3000); ignored by AStar-BMao\'", 0, &app_max_iter);
	auto method_option = op.add<Value<string>>("", "method", "\'method: AStar-BMao (exact, no iter cap) | App-BMao (approximate, uses --app_max_iter)\'", "AStar-BMao", &method);

	op.parse(argc, argv);

	if(help_option->is_set()||argc == 1) cout << op << endl;
	if(!database_option->is_set()||!query_option->is_set()) {
		printf("!!! Database file name or query file name is not provided! Exit !!!\n");
		return 0;
	}
	string database = database_option->value();
	string query = query_option->value();

	// ---- 解析 method -> 有效 app_max_iter ----
	//   AStar-BMao: 精确, 强制 app_max_iter = 0 (无上限)
	//   App-BMao:   近似, 用 --app_max_iter; 若用户没给(=0), 默认 3000
	if(method == "AStar-BMao") {
		app_max_iter = 0;
	} else if(method == "App-BMao") {
		if(app_max_iter <= 0) app_max_iter = 3000;
	} else {
		printf("!!! Unknown --method '%s' (use AStar-BMao | App-BMao). Exit !!!\n", method.c_str());
		return 0;
	}
	printf("*** method=%s, app_max_iter=%lld ***\n", method.c_str(), app_max_iter);

	vector<Graph *> db;
	map<string, ui> vM, eM;
	ui max_db_n = load_db(database.c_str(), db, vM, eM);

	printf("*** %s %s %s %d: %s %s", mode.c_str(), paradigm.c_str(), lower_bound.c_str(), threshold, database.c_str(), query.c_str());
#ifdef _EXPAND_ALL_
	//printf(" Expand_all");
#else
	printf(" Expand_one");
#endif
#ifndef _EARLY_STOP_
	printf(" NoEarly_stop");
#endif
#ifndef _UPPER_BOUND_
	printf(" NoUpper_bound");
#endif
	printf(" ***\n");

	vector<Graph *> queries;
	ui max_query_n = load_db(query.c_str(), queries, vM, eM);

	ui verify_upper_bound;
	if(threshold < 0) verify_upper_bound = INF;
	else verify_upper_bound = (ui)threshold;

	long long search_space = 0;
	long long results_cnt = 0, candidates_cnt = 0;

	ui pre = 1000;

	int *vlabel_cnt = new int[vM.size()];
	int *elabel_cnt = new int[eM.size()];
	memset(vlabel_cnt, 0, sizeof(int)*vM.size());
	memset(elabel_cnt, 0, sizeof(int)*eM.size());

	if(max_query_n > max_db_n) max_db_n = max_query_n;
	int *degree_q = new int[max_db_n];
	int *degree_g = new int[max_db_n];
	int *tmp = new int[max_db_n];

	if(strcmp(mode.c_str(), "pair") != 0&&strcmp(mode.c_str(), "search") != 0) {
		printf("!!! Wrong mode (pair | search) selection!\n");
		return 0;
	}
	if(strcmp(paradigm.c_str(), "astar") != 0&&strcmp(paradigm.c_str(), "dfs") != 0) {
		printf("!!! Wrong algorithm (astar | dfs) selection!\n");
		return 0;
	}

	Timer t;

	if(strcmp(mode.c_str(), "pair") == 0) {
		long long time1 = 0, cnt1 = 0, ss1 = 0;
		long long time2 = 0, cnt2 = 0, ss2 = 0;

		if(queries.size() != db.size()) {
			printf("Query size != db size in the pair mode\n");
			exit(0);
		}
		if(print_ged) printf("*** GEDs ***\n");
		ui min_ged = 1000000000, max_ged = 0;
		for(ui i = 0;i < queries.size();i ++) {
			ui current = i*100/queries.size();
			if(current != pre) {
				fprintf(stderr, "\r[%d%% finished]", current);
				fflush(stderr);
				//cout<<"\r["<<current<<"% finished]"<<flush;
				pre = current;
			}

			ui lb = queries[i]->ged_lower_bound_filter(db[i], verify_upper_bound, vlabel_cnt, elabel_cnt, degree_q, degree_g, tmp);
			if(lb > verify_upper_bound) continue;

			++ candidates_cnt;
			Timer t1;
			Application *app = new Application(verify_upper_bound, lower_bound.c_str(), app_max_iter);
			app->init(db[i], queries[i]);
			int res = INF;
			if(strcmp(paradigm.c_str(), "astar") == 0) res = app->AStar();
			else res = app->DFS(NULL);
#ifndef NDEBUG
			assert(res == app->compute_ged_of_BX());
#endif
			search_space += app->get_search_space();
			if(res <= verify_upper_bound) ++ results_cnt;
			else res = -1;

			if(print_ged) {
				printf("%d\n", res);
				if(res > max_ged) max_ged = res;
				if(res < min_ged) min_ged = res;
			}

			if(res == -1) {
				time2 += t1.elapsed();
				ss2 += app->get_search_space();
				++ cnt2;
			}
			else {
				time1 += t1.elapsed();
				ss1 += app->get_search_space();
				++ cnt1;
			}

			//printf("%u %u\n", db[i]->n, queries[i]->n);
			//if(db[i]->id.compare(queries[i]->id) < 0) printf("\t(pair_%u %s %s) GED: %d, Time: %s, Search space: %lld\n", i, db[i]->id.c_str(), queries[i]->id.c_str(), res, Utility::integer_to_string(t1.elapsed()).c_str(), app->get_search_space());
			//else printf("\t(pair_%u %s %s) GED: %d, Time: %s, Search space: %lld\n", i, queries[i]->id.c_str(), db[i]->id.c_str(), res, Utility::integer_to_string(t1.elapsed()).c_str(), app->get_search_space());
			//fflush(stdout);

			delete app;
		}
		fprintf(stderr, "\n");
		if(print_ged) {
			printf("*** GEDs ***\n");
			printf("min_ged: %u, max_ged: %u\n", min_ged, max_ged);
		}

		//printf("%d %d\n", cnt1, cnt2);
		if(cnt1 + cnt2 != 0) printf("total average time: %s, total average_ss: %lld\n", Utility::integer_to_string((time1+time2)/(cnt1+cnt2)).c_str(), (ss1+ss2)/(cnt1+cnt2));
		if(verify_upper_bound < INF) {
			printf("Dissimilar (%lld pairs) average time: ", cnt2);
			if(cnt2 == 0) printf("0, ");
			else printf("%s, ", Utility::integer_to_string(time2/cnt2).c_str());

			printf("Dissimilar average space: ");
			if(cnt2 == 0) printf("0\n");
			else printf("%lld\n", ss2/cnt2);

			printf("Similar (%lld pairs) average time: ", cnt1);
			if(cnt1 == 0) printf("0, ");
			else printf("%s, ", Utility::integer_to_string(time1/cnt1).c_str());
			printf("Similar average space: ");
			if(cnt1 == 0) printf("0\n");
			else printf("%lld\n", ss1/cnt1);
		}
	}
	else {
		// ---- 查询范围 [qs, qe] ----
		ui qs = (q_start < 0) ? 0 : (ui)q_start;
		ui qe = (q_end   < 0) ? (queries.size() == 0 ? 0 : (ui)queries.size() - 1) : (ui)q_end;
		if(qe >= queries.size()) qe = queries.size() == 0 ? 0 : (ui)queries.size() - 1;

		// ---- worker 数: 0 = auto-detect ----
		unsigned int nworkers = (num_workers <= 0)
			? std::max(1u, std::thread::hardware_concurrency())
			: (unsigned int)num_workers;

		ui min_ged = 1000000000, max_ged = 0;
		// 每个 query 一行的 GED 结果(供 -g 按顺序打印); 仅在 print_ged 时填充
		vector<vector<int> > per_query_ged(queries.size());
		// 每个 query 找到的 db id (res <= tau); 供 recall/precision 计算
		bool compute_recall = gt_option->is_set();
		vector<vector<int> > per_query_found(queries.size());
		// 累加每个 query 的处理时间(扫全部 db); 用于报告平均单 query 耗时(与并行无关)
		long long query_time_sum = 0;

		if(nworkers <= 1) {
			// ---------- 串行(保留原行为) ----------
			if(print_ged) printf("*** GEDs ***\n");
			for(ui i = qs;i <= qe;i ++) {
				Timer qt;
				for(ui j = 0; j < db.size();j ++) {
					ui lb = queries[i]->ged_lower_bound_filter(db[j], verify_upper_bound, vlabel_cnt, elabel_cnt, degree_q, degree_g, tmp);
					if(lb > verify_upper_bound) continue;

					++ candidates_cnt;
					Application *app = new Application(verify_upper_bound, lower_bound.c_str(), app_max_iter);
					app->init(db[j], queries[i]);
					int res = INF;
					if(strcmp(paradigm.c_str(), "astar") == 0) res = app->AStar();
					else res = app->DFS(NULL);
#ifndef NDEBUG
					assert(res == app->compute_ged_of_BX());
#endif
					if(print_ged) {
						per_query_ged[i].pb(res);
						if((ui)res > max_ged) max_ged = res;
						if((ui)res < min_ged) min_ged = res;
					}
					search_space += app->get_search_space();
					if(res <= (int)verify_upper_bound) {
						++ results_cnt;
						if(compute_recall) per_query_found[i].pb(atoi(db[j]->id.c_str()));
					}
					delete app;
				}
				query_time_sum += qt.elapsed();
			}
		}
		else {
			// ---------- 并行: 动态任务队列(每 worker 抢一个 query, 扫全部 db) ----------
			std::atomic<unsigned int> next_q((unsigned int)qs);
			std::atomic<long long> g_candidates(0), g_results(0), g_search_space(0), g_query_time(0);
			std::mutex ged_mutex;

			auto worker = [&]() {
				// 每线程独立的 scratch 数组(filter 会改写它们)
				int *t_vlc = new int[vM.size()];
				int *t_elc = new int[eM.size()];
				int *t_dq  = new int[max_db_n];
				int *t_dg  = new int[max_db_n];
				int *t_tmp = new int[max_db_n];
				memset(t_vlc, 0, sizeof(int)*vM.size());
				memset(t_elc, 0, sizeof(int)*eM.size());

				long long l_cand = 0, l_res = 0, l_ss = 0, l_qtime = 0;
				ui l_min = 1000000000, l_max = 0;

				while(true) {
					unsigned int i = next_q.fetch_add(1, std::memory_order_relaxed);
					if(i > qe) break;

					Timer qt;
					for(ui j = 0; j < db.size();j ++) {
						ui lb = queries[i]->ged_lower_bound_filter(db[j], verify_upper_bound, t_vlc, t_elc, t_dq, t_dg, t_tmp);
						if(lb > verify_upper_bound) continue;

						++ l_cand;
						Application *app = new Application(verify_upper_bound, lower_bound.c_str(), app_max_iter);
						app->init(db[j], queries[i]);
						int res = INF;
						if(strcmp(paradigm.c_str(), "astar") == 0) res = app->AStar();
						else res = app->DFS(NULL);
						if(print_ged) {
							per_query_ged[i].pb(res); // 每个 query 独占一行, 无需锁
							if((ui)res > l_max) l_max = res;
							if((ui)res < l_min) l_min = res;
						}
						l_ss += app->get_search_space();
						if(res <= (int)verify_upper_bound) {
							++ l_res;
							if(compute_recall) per_query_found[i].pb(atoi(db[j]->id.c_str())); // query i 独占, 无需锁
						}
						delete app;
					}
					l_qtime += qt.elapsed();
				}

				g_candidates.fetch_add(l_cand, std::memory_order_relaxed);
				g_results.fetch_add(l_res, std::memory_order_relaxed);
				g_search_space.fetch_add(l_ss, std::memory_order_relaxed);
				g_query_time.fetch_add(l_qtime, std::memory_order_relaxed);
				if(print_ged) {
					std::lock_guard<std::mutex> lk(ged_mutex);
					if(l_max > max_ged) max_ged = l_max;
					if(l_min < min_ged) min_ged = l_min;
				}

				delete[] t_vlc; delete[] t_elc; delete[] t_dq; delete[] t_dg; delete[] t_tmp;
			};

			vector<std::thread> pool;
			for(unsigned int w = 0; w < nworkers; w ++) pool.emplace_back(worker);
			for(auto &th : pool) th.join();

			candidates_cnt += g_candidates.load();
			results_cnt    += g_results.load();
			search_space   += g_search_space.load();
			query_time_sum += g_query_time.load();
		}

		fprintf(stderr, "\n");

		// 平均单 query 耗时(扫全部 db 的计算时间, 与并行/线程数无关)
		{
			long long nq = (long long)qe - (long long)qs + 1;
			if(nq > 0) {
				printf("*** Per-query time: avg %s us/query over %lld queries (sum of per-query latency, not divided by thread count) ***\n",
				       Utility::integer_to_string(query_time_sum / nq).c_str(), nq);
			}
		}

		if(print_ged) {
			printf("*** GEDs ***\n");
			for(ui i = qs;i <= qe;i ++) {
				for(ui k = 0;k < per_query_ged[i].size();k ++) {
					if(k) printf(" ");
					printf("%d", per_query_ged[i][k]);
				}
				printf("\n");
			}
			printf("*** GEDs ***\n");
			printf("min_ged: %u, max_ged: %u\n", min_ged, max_ged);
		}

		// ---- recall / precision (对比 Gisma ground truth) ----
		if(compute_recall) {
			if(threshold < 0) {
				printf("!!! --ground_truth needs a threshold -t for recall; skipped.\n");
			} else {
				map<int, set<int> > gt;
				load_ground_truth_within(gt_option->value().c_str(), (double)threshold, gt);

				double sum_recall = 0, sum_prec = 0, sum_iou = 0;
				int n_with_gt = 0;       // 有 ground truth (>=1 真答案) 的 query 数 (recall/IoU 分母)
				int n_with_results = 0;  // 返回了结果 (found>=1) 的 query 数 (precision 分母)
				long long tot_tp = 0, tot_gt = 0, tot_found = 0;
				for(ui i = qs;i <= qe;i ++) {
					auto it = gt.find((int)i);
					ui gt_size = (it == gt.end()) ? 0 : (ui)it->second.size();
					set<int> found(per_query_found[i].begin(), per_query_found[i].end());
					ui found_size = (ui)found.size();

					ui tp = 0;
					if(gt_size > 0) {
						for(int id : it->second) if(found.count(id)) ++tp;
					}
					tot_tp += tp; tot_gt += gt_size; tot_found += found_size;

					// recall / IoU: 只统计有 ground truth 的 query (GT=0 排除)
					if(gt_size > 0) {
						ui uni = gt_size + found_size - tp;  // |GT ∪ Found|
						sum_recall += (double)tp / gt_size;
						sum_iou    += (uni > 0) ? (double)tp / uni : 0.0;
						++ n_with_gt;
					}
					// precision: 只统计返回了结果的 query (Found=0 排除), 与 Gisma 一致
					if(found_size > 0) {
						// 注意: precision 的 TP 是相对于该 query 的真答案集; 若该 query 无 GT, tp=0
						ui tp_prec = 0;
						if(gt_size > 0) tp_prec = tp;
						sum_prec += (double)tp_prec / found_size;
						++ n_with_results;
					}
				}

				printf("*** Recall/Precision/IoU (threshold=%d) ***\n", threshold);
				printf("queries with GT: %d (recall/IoU denom), queries with results: %d (precision denom)\n", n_with_gt, n_with_results);
				long long tot_uni = tot_gt + tot_found - tot_tp;
				printf("Avg recall: %.2f%%, Avg precision: %.2f%%, Avg IoU: %.2f%%\n",
				       n_with_gt > 0 ? 100.0*sum_recall/n_with_gt : 0.0,
				       n_with_results > 0 ? 100.0*sum_prec/n_with_results : 0.0,
				       n_with_gt > 0 ? 100.0*sum_iou/n_with_gt : 0.0);
				printf("Micro recall: %.2f%% (TP=%lld / GT=%lld), Micro precision: %.2f%% (TP=%lld / Found=%lld), Micro IoU: %.2f%% (TP=%lld / Union=%lld)\n",
				       tot_gt ? 100.0*tot_tp/tot_gt : 0.0, tot_tp, tot_gt,
				       tot_found ? 100.0*tot_tp/tot_found : 0.0, tot_tp, tot_found,
				       tot_uni ? 100.0*tot_tp/tot_uni : 0.0, tot_tp, tot_uni);
			}
		}
	}
	printf("Total time: %s (microseconds), total search space: %lld\n #candidates: %lld, #matches: %lld\n", Utility::integer_to_string(t.elapsed()).c_str(), search_space, candidates_cnt, results_cnt);
	fflush(stdout);

	delete[] vlabel_cnt; vlabel_cnt = NULL;
	delete[] elabel_cnt; elabel_cnt = NULL;
	delete[] degree_q; degree_q = NULL;
	delete[] degree_g; degree_g = NULL;
	delete[] tmp; tmp = NULL;

	for(ui i = 0;i < db.size();i ++) {
		delete db[i];
		db[i] = nullptr;
	}
	for(ui i = 0;i < queries.size();i ++) {
		delete queries[i];
		queries[i] = nullptr;
	}
	return 0;
}
