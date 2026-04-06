#!/usr/bin/env python3
"""
Gisma GED Calculator for LAN (Final Version)
Computes real GED using Gisma's AppForComputation method.
Invokes the Gisma C++ program through a Python wrapper.

Fallback strategy (strictly following LAN paper Section VII):
1. First attempt exact GED (10-second timeout)
2. On timeout, use min(VJ, Hung, Beam) from three approximate algorithms
   - VJ [56]: Volgenant-Jonker algorithm (lapjv)
   - Hung [57]: Hungarian algorithm (scipy)
   - Beam [58]: A* Beam Search algorithm (Neuhaus et al. 2006)

References:
[56] Fankhauser et al. - Speeding up graph edit distance computation through fast bipartite matching
[57] Riesen & Bunke - Approximate graph edit distance computation by means of bipartite graph matching
[58] Neuhaus, Riesen, Bunke - Fast suboptimal algorithms for the computation of graph edit distance (SSPR 2006)
"""

import networkx as nx
import numpy as np
import tempfile
import os
import logging
import subprocess
import re
from typing import Dict, Optional, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check available approximate GED algorithms
SCIPY_AVAILABLE = False
LAPJV_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not installed. Hungarian algorithm unavailable.")

try:
    import lapjv
    LAPJV_AVAILABLE = True
except ImportError:
    logger.warning("lapjv not installed. VJ algorithm unavailable.")
    logger.warning("Install with: pip install lapjv")


class GismaGEDCalculator:
    """
    Computes GED using the Gisma C++ program.
    Calls AppForComputation method (ub set to inf for exact GED).

    Fallback strategy (strictly following LAN paper Section VII):
    1. First attempt exact GED (timeout seconds)
    2. On timeout, use min(VJ, Hung, Beam) from three approximate algorithms
       - VJ: Volgenant-Jonker bipartite matching
       - Hung: Hungarian bipartite matching
       - Beam: Beam search tree search
    """

    def __init__(self,
                 gisma_exe: str = None,
                 dataset_path: str = None,
                 embedding_file: str = None,
                 use_app_for_computation: bool = True,
                 timeout: int = 10,
                 beam_width: int = 100,
                 ged_upper_bound: int = None):
        """
        Args:
            gisma_exe: Path to the Gisma executable
            dataset_path: Dataset path (containing db.txt etc.)
            embedding_file: GREED embedding file path (no longer used for fallback, kept for compatibility)
            use_app_for_computation: Whether to use AppForComputation (recommended)
            timeout: Computation timeout in seconds, default 10s (per LAN paper)
            beam_width: Beam search beam width (default 100)
            ged_upper_bound: GED upper bound (for A* pruning, None = no bound)
        """
        # Use relative paths based on script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gisma_base = os.path.join(script_dir, '..', 'Gisma')

        if gisma_exe is None:
            # Try multiple possible paths
            possible_paths = [
                os.path.join(gisma_base, 'build', 'GismaProject'),           # Linux direct
                os.path.join(gisma_base, 'build', 'Release', 'GismaProject'), # Linux Release
                os.path.join(gisma_base, 'build', 'Debug', 'GismaProject'),   # Linux Debug
                os.path.join(gisma_base, 'build', 'Release', 'GismaProject.exe'), # Windows Release
                os.path.join(gisma_base, 'build', 'Debug', 'GismaProject.exe'),   # Windows Debug
            ]
            gisma_exe = None
            for path in possible_paths:
                if os.path.exists(path):
                    gisma_exe = path
                    break
            if gisma_exe is None:
                # Default fallback (will fail later with clear error)
                gisma_exe = possible_paths[0]

        if dataset_path is None:
            dataset_path = os.path.join(gisma_base, 'datasets', 'AIDS')

        # Extract dataset name from path and auto-detect embedding file
        if embedding_file is None:
            # Get dataset name from path (e.g., "AIDS" from ".../datasets/AIDS")
            dataset_name = os.path.basename(dataset_path.rstrip('/\\'))
            embedding_file = os.path.join(gisma_base, 'embeddings', dataset_name, f'{dataset_name}_embeddings.bin')

        self.gisma_exe = gisma_exe
        self.dataset_path = dataset_path
        self.db_file = os.path.join(dataset_path, "db.txt")
        self.embedding_file = embedding_file
        self.use_app_for_computation = use_app_for_computation
        self.timeout = timeout
        self.beam_width = beam_width
        self.ged_upper_bound = ged_upper_bound  # GED upper bound (for A* pruning)
        self.ged_source_cache = {}  # Records the source of each GED: 'exact', 'approx', 'heuristic'

        # Query embeddings (optional, for external queries not in database)
        self.query_embeddings = None  # Will be set externally if needed
        self.current_query_emb = None  # Current query embedding for fallback

        # Check approximate GED algorithm availability
        self.approx_methods_available = []
        if SCIPY_AVAILABLE:
            self.approx_methods_available.append('hung')
        if LAPJV_AVAILABLE:
            self.approx_methods_available.append('vj')
        # Beam search (Neuhaus 2006) is self-implemented, always available
        self.approx_methods_available.append('beam')

        # Statistics
        self.stats = {
            'total_calls': 0,
            'ged_success': 0,
            'ged_timeout': 0,
            'ged_exception': 0,
            'approx_fallback': 0  # Number of times min(VJ, Hung, Beam) was used
        }

        # Verify files exist
        if not os.path.exists(self.gisma_exe):
            raise FileNotFoundError(f"Gisma executable not found: {self.gisma_exe}")
        if not os.path.exists(self.db_file):
            raise FileNotFoundError(f"Database file not found: {self.db_file}")

        # Load all graph data into memory
        self.all_graphs = self._load_all_graphs()

        # Load GREED embedding vectors
        self.embeddings = None
        self.id_to_index = {}
        if os.path.exists(self.embedding_file):
            self._load_embeddings()
            logger.info(f"  Loaded embeddings: {self.embeddings.shape}")
        else:
            logger.warning(f"  Embedding file not found: {self.embedding_file}")
            logger.warning(f"  Will use heuristic GED on timeout")

        logger.info(f"Gisma GED Calculator initialized")
        logger.info(f"  Executable: {self.gisma_exe}")
        logger.info(f"  Database: {self.db_file}")
        logger.info(f"  Loaded {len(self.all_graphs)} graphs")
        logger.info(f"  Method: {'AppForComputation' if use_app_for_computation else 'AStar (0.1s timeout)'}")
        logger.info(f"  Timeout: {self.timeout}s")
        # Fallback strategy (per LAN paper): min(VJ, Hung, Beam)
        if self.approx_methods_available:
            methods_str = ', '.join(self.approx_methods_available)
            logger.info(f"  Fallback: min({methods_str}) - LAN paper style")
        else:
            logger.info(f"  Fallback: Heuristic GED (no approx methods available)")

    def _load_embeddings(self):
        """Load GREED embedding vectors (GREED format: ii header + [id + emb] x N)"""
        import struct

        file_size = os.path.getsize(self.embedding_file)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"  Loading embeddings from {self.embedding_file} ({file_size_mb:.1f} MB)")

        with open(self.embedding_file, 'rb') as f:
            # Read header info (GREED format: 2 ints)
            num_graphs, embedding_dim = struct.unpack('ii', f.read(8))
            logger.info(f"    {num_graphs} embeddings, dim={embedding_dim}")

            # Read all embedding vectors (each entry: graph_id + embedding)
            embeddings_list = []
            for i in tqdm(range(num_graphs), desc="    Loading embeddings", unit="emb"):
                graph_id = struct.unpack('i', f.read(4))[0]
                emb = np.frombuffer(f.read(embedding_dim * 4), dtype=np.float32).copy()
                embeddings_list.append(emb)

            # Convert to numpy array
            self.embeddings = np.array(embeddings_list, dtype=np.float32)

            # Build ID-to-index mapping
            for i, graph_id in enumerate(sorted(self.all_graphs.keys(), key=int)):
                self.id_to_index[graph_id] = i

    def _load_all_graphs(self) -> Dict[str, nx.Graph]:
        """Load all graph data"""
        graphs = {}
        current_graph = None
        graph_count = 0

        # Get file size for progress
        file_size = os.path.getsize(self.db_file)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"  Loading graphs from {self.db_file} ({file_size_mb:.1f} MB)")

        # Count total lines for progress bar
        with open(self.db_file, 'r') as f:
            total_lines = sum(1 for _ in f)

        with open(self.db_file, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="    Loading graphs", unit="line"):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                if parts[0] == 'ID':
                    if current_graph is not None:
                        graphs[current_graph.graph['id']] = current_graph
                        graph_count += 1
                    graph_id = parts[1]
                    current_graph = nx.Graph(id=graph_id)

                elif parts[0] == 'v' and current_graph is not None:
                    node_id = int(parts[1])
                    label = parts[2] if len(parts) > 2 else "0"
                    current_graph.add_node(node_id, label=label)

                elif parts[0] == 'e' and current_graph is not None:
                    src = int(parts[1])
                    dst = int(parts[2])
                    current_graph.add_edge(src, dst)

        if current_graph is not None:
            graphs[current_graph.graph['id']] = current_graph
            graph_count += 1

        logger.info(f"    Loaded {graph_count} graphs.")

        return graphs

    def compute_ged(self, q: nx.Graph, g: nx.Graph, q_emb=None) -> float:
        """
        Compute GED between two graphs.

        Args:
            q: Query graph
            g: Database graph
            q_emb: Query embedding vector (optional, if query is not in the database)

        Returns:
            GED value
        """
        qid = q.graph.get("id")
        gid = g.graph.get("id")

        # Check if they are the SAME graph object (not just same ID)
        if q is g:
            return 0.0

        # Use current_query_emb if q_emb not provided
        if q_emb is None and self.current_query_emb is not None:
            q_emb = self.current_query_emb

        # Compute GED
        # When timeout <= 0, skip exact GED and use fallback directly
        if self.timeout <= 0:
            ged, source = self._fallback_ged(q, g, qid, gid, q_emb)
        else:
            source = 'exact'  # Default assumption is exact GED
            try:
                ged, source = self._compute_ged_via_gisma(q, g, q_emb)
            except Exception as e:
                logger.warning(f"Gisma GED computation failed: {e}")
                logger.warning("Falling back to min(VJ, Hung, Beam)")
                ged, source = self._fallback_ged(q, g, qid, gid, q_emb)

        # Record source
        cache_key = (qid, gid)
        self.ged_source_cache[cache_key] = source
        self.ged_source_cache[(gid, qid)] = source

        return ged

    def get_ged_source(self, q: nx.Graph, g: nx.Graph) -> str:
        """Get the source of a GED computation"""
        qid = q.graph.get("id")
        gid = g.graph.get("id")
        cache_key = (qid, gid)
        return self.ged_source_cache.get(cache_key, 'unknown')

    def _compute_ged_via_gisma(self, q: nx.Graph, g: nx.Graph, q_emb=None):
        """
        Compute GED via the Gisma program.
        Uses the newly added compute_ged mode.
        On timeout, uses GREED embeddings as fallback.

        Args:
            q: Query graph object
            g: Target graph object
            q_emb: Query embedding vector (optional, for fallback)

        Returns:
            (ged, source): GED value and source label ('exact' or 'embedding')
        """
        self.stats['total_calls'] += 1

        qid = q.graph.get("id")
        gid = g.graph.get("id")

        query_file = None
        target_file = None

        try:
            # Create temporary files to save graph data
            with tempfile.NamedTemporaryFile(mode='w', suffix='_q.txt', delete=False) as qf:
                qf.write(f"ID {qid}\n")
                for node in q.nodes():
                    label = q.nodes[node].get('label', '0')
                    qf.write(f"v {node} {label}\n")
                for src, dst in q.edges():
                    qf.write(f"e {src} {dst} 0\n")
                query_file = qf.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='_g.txt', delete=False) as gf:
                gf.write(f"ID {gid}\n")
                for node in g.nodes():
                    label = g.nodes[node].get('label', '0')
                    gf.write(f"v {node} {label}\n")
                for src, dst in g.edges():
                    gf.write(f"e {src} {dst} 0\n")
                target_file = gf.name

            # Call Gisma's compute_ged mode
            # Use AppForComputation (exact GED), not App (approximate, has early termination bug)
            cmd = [
                self.gisma_exe,
                "--mode", "compute_ged",
                "--query_file", query_file,
                "--target_file", target_file,
                "--ged_algorithm", "AppForComputation",
                "--tau_search", str(self.ged_upper_bound if self.ged_upper_bound is not None else 10000000),
                "--app_max_iter", "1000000"  # Maximum iterations
            ]

            logger.debug(f"GED command: {' '.join(cmd)}")  # Debug: print command

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Parse output
            ged = self._parse_ged_output(result.stdout)

            # Check for timeout (inf indicates timeout or parse failure)
            if ged == float('inf'):
                self.stats['ged_timeout'] += 1
                # Fallback order: Java toolkit > GREED embeddings > Heuristic
                ged, source = self._fallback_ged(q, g, qid, gid, q_emb)
            else:
                self.stats['ged_success'] += 1
                logger.info(f"GED({qid}, {gid}) = {ged:.2f} [exact]")
                source = 'exact'

            return ged, source

        except subprocess.TimeoutExpired:
            logger.debug(f"Process timeout for {qid} vs {gid}")
            self.stats['ged_timeout'] += 1
            # Fallback
            ged, source = self._fallback_ged(q, g, qid, gid, q_emb)
            return ged, source

        except Exception as e:
            logger.warning(f"Error computing GED: {e}")
            self.stats['ged_exception'] += 1
            # Fallback
            ged, source = self._fallback_ged(q, g, qid, gid, q_emb)
            return ged, source

        finally:
            # Safely clean up temporary files
            import time
            for temp_file in [query_file, target_file]:
                if temp_file and os.path.exists(temp_file):
                    for attempt in range(3):
                        try:
                            os.unlink(temp_file)
                            break
                        except (PermissionError, OSError):
                            # Windows may need to wait for file handle release
                            time.sleep(0.1)
                        except Exception:
                            break

    def _fallback_ged(self, q: nx.Graph, g: nx.Graph, qid: str, gid: str, q_emb=None):
        """
        Fallback strategy when GED computation fails (strictly following LAN paper Section VII).

        Uses min(VJ, Hung, Beam) from three approximate algorithms:
        - VJ [56]: Volgenant-Jonker bipartite matching (lapjv)
        - Hung [57]: Hungarian bipartite matching (scipy)
        - Beam [58]: Beam search tree search (gklearn)

        Args:
            q: Query graph
            g: Target graph
            qid: Query graph ID
            gid: Target graph ID
            q_emb: Query embedding vector (optional, no longer used for fallback)

        Returns:
            (ged, source): GED value and source label ('approx' or 'heuristic')
        """
        # Per LAN paper, use min(VJ, Hung, Beam)
        results = []
        methods_used = []

        # 1. VJ (Volgenant-Jonker) algorithm
        if 'vj' in self.approx_methods_available:
            try:
                vj_ged = self._compute_vj_ged(q, g)
                if vj_ged is not None and vj_ged < float('inf'):
                    results.append(vj_ged)
                    methods_used.append(f'VJ={vj_ged:.2f}')
            except Exception as e:
                logger.debug(f"VJ GED failed: {e}")

        # 2. Hungarian algorithm
        if 'hung' in self.approx_methods_available:
            try:
                hung_ged = self._compute_hungarian_ged(q, g)
                if hung_ged is not None and hung_ged < float('inf'):
                    results.append(hung_ged)
                    methods_used.append(f'Hung={hung_ged:.2f}')
            except Exception as e:
                logger.debug(f"Hungarian GED failed: {e}")

        # 3. Beam search algorithm (Neuhaus 2006)
        if 'beam' in self.approx_methods_available:
            try:
                beam_ged = self._compute_beam_ged(q, g, beam_width=self.beam_width)
                if beam_ged is not None and beam_ged < float('inf'):
                    results.append(beam_ged)
                    methods_used.append(f'Beam={beam_ged:.2f}')
            except Exception as e:
                logger.debug(f"Beam GED failed: {e}")

        # Take the minimum
        min_ged = min(results)
        self.stats['approx_fallback'] += 1
        logger.info(f"GED({qid}, {gid}) = {min_ged:.2f} [approx: {', '.join(methods_used)}]")
        return min_ged, 'approx'

    def _build_cost_matrix(self, q: nx.Graph, g: nx.Graph) -> np.ndarray:
        """
        Build LSAPE cost matrix (for VJ and Hungarian algorithms).

        Strictly follows Riesen & Bunke 2009 paper Definition 4 (p.954):
        "Approximate graph edit distance computation by means of bipartite graph matching"

        Cost matrix size is (n+m) x (n+m), where n = |V(q)|, m = |V(g)|
        - Top-left n x m: Node substitution cost c(u_i -> v_j) = node label cost + edge edit cost lower bound
          * Edge edit cost obtained by bipartite matching on adjacent edges of u_i and v_j
        - Top-right n x n: Node deletion cost (diagonal) c(u_i -> epsilon)
        - Bottom-left m x m: Node insertion cost (diagonal) c(epsilon -> v_j)
        - Bottom-right: 0 (dummy to dummy)

        Args:
            q: Query graph
            g: Target graph

        Returns:
            Cost matrix (n+m) x (n+m)
        """
        n = q.number_of_nodes()
        m = g.number_of_nodes()

        # Initialize cost matrix
        cost = np.full((n + m, n + m), np.inf)

        # Node lists (ensure consistent ordering)
        q_nodes = list(q.nodes())
        g_nodes = list(g.nodes())

        # Node substitution cost (top-left n x m)
        # Per Riesen & Bunke Definition 4:
        # c(u_i -> v_j) = c_node(u_i -> v_j) + min_edge_cost(E_ui, E_vj)
        # where min_edge_cost is obtained by bipartite matching on adjacent edges
        for i, u in enumerate(q_nodes):
            u_label = q.nodes[u].get('label', '0')
            u_edges = list(q.edges(u))  # Adjacent edges of u

            for j, v in enumerate(g_nodes):
                v_label = g.nodes[v].get('label', '0')
                v_edges = list(g.edges(v))  # Adjacent edges of v

                # 1. Node label substitution cost
                node_cost = 0.0 if u_label == v_label else 1.0

                # 2. Edge edit cost: bipartite matching on adjacent edges
                edge_cost = self._compute_edge_edit_cost(q, g, u, v, u_edges, v_edges)

                cost[i, j] = node_cost + edge_cost

        # Node deletion cost (top-right n x n diagonal)
        # c(u_i -> epsilon) = c_node(u_i -> epsilon) + sum(c_edge(e -> epsilon)) for all edges incident to u_i
        # Each edge deletion costs 1, but each edge is counted at both endpoints, so divide by 2
        for i, u in enumerate(q_nodes):
            cost[i, m + i] = 1.0 + q.degree(u) / 2.0

        # Node insertion cost (bottom-left m x m diagonal)
        # c(epsilon -> v_j) = c_node(epsilon -> v_j) + sum(c_edge(epsilon -> e)) for all edges incident to v_j
        for j, v in enumerate(g_nodes):
            cost[n + j, j] = 1.0 + g.degree(v) / 2.0

        # Bottom-right (dummy to dummy) = 0
        cost[n:, m:] = 0

        return cost

    def _compute_edge_edit_cost(self, q: nx.Graph, g: nx.Graph,
                                 u: int, v: int,
                                 u_edges: list, v_edges: list) -> float:
        """
        Compute the edge edit cost lower bound when mapping node u to node v.

        Strictly follows Riesen & Bunke 2009 paper:
        Builds a cost matrix from u's adjacent edge set E_u and v's adjacent edge set E_v,
        then solves the minimum cost matching using the Hungarian algorithm.

        Edge cost matrix (p + q) x (p + q), where p = |E_u|, q = |E_v|:
        - Top-left p x q: Edge substitution cost (1 if labels differ, 0 if same)
        - Top-right p x p: Edge deletion cost (diagonal = 1)
        - Bottom-left q x q: Edge insertion cost (diagonal = 1)
        - Bottom-right: 0

        Note: Each edge is counted at both endpoints, so the final result is divided by 2.

        Args:
            q: Query graph
            g: Target graph
            u: Node in q
            v: Node in g
            u_edges: List of u's adjacent edges
            v_edges: List of v's adjacent edges

        Returns:
            Edge edit cost lower bound (already divided by 2)
        """
        p = len(u_edges)
        q_count = len(v_edges)

        # Boundary cases
        if p == 0 and q_count == 0:
            return 0.0
        if p == 0:
            # Only insertions, each edge costs 1
            return q_count / 2.0
        if q_count == 0:
            # Only deletions, each edge costs 1
            return p / 2.0

        # Build edge cost matrix (p + q_count) x (p + q_count)
        edge_cost = np.full((p + q_count, p + q_count), np.inf)

        # Top-left p x q_count: Edge substitution cost
        # Note: We only consider edge labels here, not the other endpoint
        # (because when building the node cost matrix, we don't yet know other node mappings)
        for ei, (u1, u2) in enumerate(u_edges):
            # Get edge label (if available)
            u_edge_label = q.edges[u1, u2].get('label', '0')

            for ej, (v1, v2) in enumerate(v_edges):
                v_edge_label = g.edges[v1, v2].get('label', '0')

                # Edge substitution cost: 1 if labels differ
                edge_cost[ei, ej] = 0.0 if u_edge_label == v_edge_label else 1.0

        # Top-right p x p: Edge deletion cost (diagonal = 1)
        for ei in range(p):
            edge_cost[ei, q_count + ei] = 1.0

        # Bottom-left q_count x q_count: Edge insertion cost (diagonal = 1)
        for ej in range(q_count):
            edge_cost[p + ej, ej] = 1.0

        # Bottom-right: 0
        edge_cost[p:, q_count:] = 0.0

        # Solve minimum cost matching using Hungarian algorithm
        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(edge_cost)
            min_cost = edge_cost[row_ind, col_ind].sum()
        else:
            # If scipy is unavailable, use greedy approximation
            min_cost = self._greedy_edge_matching(edge_cost, p, q_count)

        # Divide by 2 (each edge is counted at both endpoints)
        return min_cost / 2.0

    def _greedy_edge_matching(self, edge_cost: np.ndarray, p: int, q_count: int) -> float:
        """
        Greedy edge matching (fallback when scipy is unavailable)
        """
        total = p + q_count
        row_used = [False] * total
        col_used = [False] * total
        total_cost = 0.0

        # Greedily select the lowest-cost matching
        while True:
            min_cost = np.inf
            min_row = -1
            min_col = -1

            for i in range(total):
                if row_used[i]:
                    continue
                for j in range(total):
                    if col_used[j]:
                        continue
                    if edge_cost[i, j] < min_cost:
                        min_cost = edge_cost[i, j]
                        min_row = i
                        min_col = j

            if min_row == -1:
                break

            row_used[min_row] = True
            col_used[min_col] = True
            total_cost += min_cost

        return total_cost

    def _compute_induced_cost(self, q: nx.Graph, g: nx.Graph, assignment: np.ndarray) -> float:
        """
        Compute induced GED from the node mapping.

        Args:
            q: Query graph
            g: Target graph
            assignment: Node mapping array, assignment[i] = j means q's i-th node maps to g's j-th node.
                        If j >= m, it represents deletion.

        Returns:
            induced GED
        """
        n = q.number_of_nodes()
        m = g.number_of_nodes()

        q_nodes = list(q.nodes())
        g_nodes = list(g.nodes())

        cost = 0.0

        # Node mapping
        node_map = {}  # q_node -> g_node (or None for deletion)
        g_used = set()  # g nodes that have been mapped to

        for i, u in enumerate(q_nodes):
            j = assignment[i]
            if j < m:
                # Node substitution
                v = g_nodes[j]
                node_map[u] = v
                g_used.add(v)
                # Label substitution cost
                if q.nodes[u].get('label', '0') != g.nodes[v].get('label', '0'):
                    cost += 1
            else:
                # Node deletion
                node_map[u] = None
                cost += 1

        # Node insertion (unmapped nodes in g)
        for v in g_nodes:
            if v not in g_used:
                cost += 1

        # Edge edit cost
        # Delete edges: edges in q whose mapped endpoints do not exist in g
        for u1, u2 in q.edges():
            v1 = node_map.get(u1)
            v2 = node_map.get(u2)
            if v1 is None or v2 is None:
                # Endpoint was deleted, so the edge is also deleted
                cost += 1
            elif not g.has_edge(v1, v2):
                # Mapped edge does not exist in g
                cost += 1

        # Insert edges: edges in g whose endpoint preimages do not exist in q
        reverse_map = {v: u for u, v in node_map.items() if v is not None}
        for v1, v2 in g.edges():
            u1 = reverse_map.get(v1)
            u2 = reverse_map.get(v2)
            if u1 is None or u2 is None:
                # Endpoint is newly inserted, so the edge is also newly inserted
                cost += 1
            elif not q.has_edge(u1, u2):
                # Preimage edge does not exist in q
                cost += 1

        return cost

    def _compute_vj_ged(self, q: nx.Graph, g: nx.Graph) -> float:
        """
        Compute approximate GED using the Volgenant-Jonker algorithm [56].

        VJ is an efficient LSAP solver, faster than Hungarian.

        Args:
            q: Query graph
            g: Target graph

        Returns:
            Approximate GED value
        """
        if not LAPJV_AVAILABLE:
            return None

        # Build cost matrix
        cost = self._build_cost_matrix(q, g)

        # lapjv cannot handle inf, replace with large value
        cost_finite = np.where(np.isinf(cost), 1e10, cost)

        # Solve using lapjv
        # lapjv returns (row_to_col, col_to_row, (cost, u, v))
        row_to_col, _, _ = lapjv.lapjv(cost_finite)

        # Compute induced cost
        ged = self._compute_induced_cost(q, g, row_to_col)

        return ged

    def _compute_hungarian_ged(self, q: nx.Graph, g: nx.Graph) -> float:
        """
        Compute approximate GED using the Hungarian algorithm [57].

        Hungarian is a classic LSAP solver with O(n^3) complexity.

        Args:
            q: Query graph
            g: Target graph

        Returns:
            Approximate GED value
        """
        if not SCIPY_AVAILABLE:
            return None

        # Build cost matrix
        cost = self._build_cost_matrix(q, g)

        # Solve using scipy's Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build assignment array
        assignment = col_ind

        # Compute induced cost
        ged = self._compute_induced_cost(q, g, assignment)

        return ged

    def _compute_beam_ged(self, q: nx.Graph, g: nx.Graph, beam_width: int = 1) -> float:
        """
        Compute approximate GED using the A*-Beamsearch algorithm.

        Strictly follows Neuhaus, Riesen, Bunke (2006) paper:
        "Fast Suboptimal Algorithms for the Computation of Graph Edit Distance"
        SSPR/SPR 2006, LNCS 4109, pp. 163-172.

        Algorithm 1 (A*-Beamsearch):
        - OPEN set contains partial edit paths
        - Best-first expansion: always expand the path with smallest f(p) = g(p) + h(p)
        - Beam pruning: when OPEN exceeds size s, remove the worst paths
        - Return cost when a complete path is found

        Heuristic function h(p) based on Bunke & Allermann 1983 [11]

        Args:
            q: Query graph (source graph G1)
            g: Target graph (target graph G2)
            beam_width: Beam width s (paper tested 10, 100, 1000)

        Returns:
            Approximate GED value
        """
        import heapq

        n = q.number_of_nodes()
        m = g.number_of_nodes()

        # Boundary cases
        if n == 0 and m == 0:
            return 0.0
        if n == 0:
            return float(m + g.number_of_edges())  # Insert all nodes and edges
        if m == 0:
            return float(n + q.number_of_edges())  # Delete all nodes and edges

        q_nodes = list(q.nodes())
        g_nodes = list(g.nodes())

        def compute_heuristic(depth: int, used_g: frozenset) -> float:
            """
            Heuristic function h(p): lower bound estimate of remaining cost.

            Strictly follows Neuhaus 2006 paper Section 2 (p.166-167):
            Based on Bunke & Allermann 1983 [11] method.

            For remaining unprocessed nodes:
            - n1 = number of remaining q nodes
            - n2 = number of remaining g nodes
            - Compute all possible node substitution costs, take the smallest min{n1, n2}
            - Add |n1 - n2| deletion or insertion operations
            """
            # Remaining unmapped nodes
            remaining_q_nodes = q_nodes[depth:]  # q nodes starting from depth
            remaining_g_nodes = [v for v in g_nodes if v not in used_g]

            n1 = len(remaining_q_nodes)
            n2 = len(remaining_g_nodes)

            if n1 == 0 and n2 == 0:
                return 0.0

            if n1 == 0:
                # Only need to insert remaining g nodes
                return float(n2)

            if n2 == 0:
                # Only need to delete remaining q nodes
                return float(n1)

            # Compute all possible node substitution costs (only label cost as lower bound)
            substitution_costs = []
            for u in remaining_q_nodes:
                u_label = q.nodes[u].get('label', '0')
                for v in remaining_g_nodes:
                    v_label = g.nodes[v].get('label', '0')
                    cost = 0 if u_label == v_label else 1
                    substitution_costs.append(cost)

            # Sort and take the smallest min{n1, n2} substitution costs
            substitution_costs.sort()
            num_substitutions = min(n1, n2)
            min_substitution_cost = sum(substitution_costs[:num_substitutions])

            # Add deletion or insertion cost
            h = min_substitution_cost + abs(n1 - n2)

            return h

        def compute_final_cost(g_cost: float, mapping: dict, used_g: frozenset) -> float:
            """
            Compute the final cost of a complete edit path.

            Handles remaining unmapped g nodes (insertion operations).
            """
            final_cost = g_cost

            # Insert unmapped g nodes
            unused_g = [v for v in g_nodes if v not in used_g]
            final_cost += len(unused_g)  # Node insertion cost

            # Insert edges between these new nodes
            for i, v1 in enumerate(unused_g):
                for v2 in unused_g[i+1:]:
                    if g.has_edge(v1, v2):
                        final_cost += 1

                # Insert edges between new nodes and already-mapped nodes
                for mapped_v in used_g:
                    if g.has_edge(v1, mapped_v):
                        final_cost += 1

            return final_cost

        # OPEN set: min-heap, elements are (f_cost, unique_id, depth, g_cost, mapping, used_g)
        # depth: number of processed q nodes (0 to n)
        # mapping: dict, q_node -> g_node or None (deletion)
        # used_g: frozenset, g nodes that have been mapped to
        unique_counter = 0
        initial_h = compute_heuristic(0, frozenset())
        OPEN = [(initial_h, unique_counter, 0, 0.0, {}, frozenset())]
        unique_counter += 1

        while OPEN:
            # Best-first: pop the path with smallest f(p)
            f_cost, _, depth, g_cost, mapping, used_g = heapq.heappop(OPEN)

            # Check if this is a complete path (all q nodes processed)
            if depth == n:
                # Compute final cost (including insertion of remaining g nodes)
                return compute_final_cost(g_cost, mapping, used_g)

            # Expand: process the depth-th q node
            u = q_nodes[depth]

            # Option 1: Map u to an unused node in g (node substitution)
            for v in g_nodes:
                if v in used_g:
                    continue

                # Compute incremental cost
                delta_cost = 0.0

                # Node substitution cost (1 if labels differ)
                if q.nodes[u].get('label', '0') != g.nodes[v].get('label', '0'):
                    delta_cost += 1

                # Edge edit cost: check edges between u and already-mapped nodes
                for prev_u, prev_v in mapping.items():
                    if prev_v is None:
                        continue
                    q_has_edge = q.has_edge(u, prev_u)
                    g_has_edge = g.has_edge(v, prev_v)

                    if q_has_edge and not g_has_edge:
                        delta_cost += 1  # Edge deletion
                    elif not q_has_edge and g_has_edge:
                        delta_cost += 1  # Edge insertion

                new_g_cost = g_cost + delta_cost
                new_mapping = dict(mapping)
                new_mapping[u] = v
                new_used_g = used_g | {v}

                # Compute f = g + h
                new_h = compute_heuristic(depth + 1, new_used_g)
                new_f_cost = new_g_cost + new_h

                heapq.heappush(OPEN, (new_f_cost, unique_counter, depth + 1,
                                       new_g_cost, new_mapping, new_used_g))
                unique_counter += 1

                # Beam pruning: remove worst when OPEN exceeds beam_width
                if len(OPEN) > beam_width:
                    # Convert to list, sort, keep the best beam_width entries
                    OPEN = heapq.nsmallest(beam_width, OPEN)
                    heapq.heapify(OPEN)

            # Option 2: Delete u (map to epsilon)
            delta_cost = 1  # Node deletion cost

            # Delete edges between u and already-mapped nodes
            for prev_u, prev_v in mapping.items():
                if prev_v is None:
                    continue
                if q.has_edge(u, prev_u):
                    delta_cost += 1  # Edge deletion

            new_g_cost = g_cost + delta_cost
            new_mapping = dict(mapping)
            new_mapping[u] = None

            new_h = compute_heuristic(depth + 1, used_g)
            new_f_cost = new_g_cost + new_h

            heapq.heappush(OPEN, (new_f_cost, unique_counter, depth + 1,
                                   new_g_cost, new_mapping, used_g))
            unique_counter += 1

            # Beam pruning
            if len(OPEN) > beam_width:
                OPEN = heapq.nsmallest(beam_width, OPEN)
                heapq.heapify(OPEN)

        # OPEN is empty but no complete path found (should not happen)
        return float('inf')

    def _parse_ged_output(self, output: str) -> float:
        """
        Parse the output of Gisma's compute_ged mode.
        Output format:
        === GED Computation Result ===
        ...
        GED: 31
        ...
        """
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('GED:'):
                try:
                    ged_str = line.split(':')[1].strip()
                    ged_value = float(ged_str)
                    # If INF (10000000), return float('inf') indicating timeout
                    if ged_value >= 10000000:
                        logger.debug(f"GED result is INF (timeout): {ged_value}")
                        return float('inf')
                    return ged_value
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing line '{line}': {e}")
                    pass

        # If no 'GED:' line found, the program may have exited early or failed
        logger.debug(f"No 'GED:' line found in output (first 500 chars): {output[:500]}")
        return float('inf')

    def _compute_embedding_distance(self, qid: str, gid: str, q_emb=None) -> float:
        """
        Compute distance using GREED embeddings.
        Uses Euclidean distance.

        Args:
            qid: Query graph ID
            gid: Target graph ID
            q_emb: Query embedding vector (optional, if query is not in the database)
        """
        if self.embeddings is None:
            return float('inf')

        # Get target graph embedding
        gid_idx = self.id_to_index.get(gid)
        if gid_idx is None:
            return float('inf')
        g_emb = self.embeddings[gid_idx]

        # Get query embedding
        # Prefer externally set current_query_emb (for queries not in the database)
        if q_emb is not None:
            query_emb = q_emb
        elif self.current_query_emb is not None:
            query_emb = self.current_query_emb
        else:
            # Try to get from database embeddings (if query is in the database)
            qid_idx = self.id_to_index.get(qid)
            if qid_idx is not None:
                query_emb = self.embeddings[qid_idx]
            else:
                return float('inf')

        # Compute Euclidean distance
        dist = np.linalg.norm(query_emb - g_emb)
        return float(dist)

    def print_stats(self):
        """Print statistics"""
        total = self.stats['total_calls']
        if total == 0:
            logger.info("No GED computations performed yet")
            return

        logger.info("=== GED Computation Statistics ===")
        logger.info(f"Total GED calls: {total}")
        logger.info(f"Successful exact GED: {self.stats['ged_success']} ({100.0*self.stats['ged_success']/total:.1f}%)")
        logger.info(f"GED timeouts: {self.stats['ged_timeout']} ({100.0*self.stats['ged_timeout']/total:.1f}%)")
        logger.info(f"GED exceptions: {self.stats['ged_exception']} ({100.0*self.stats['ged_exception']/total:.1f}%)")
        logger.info(f"Approx fallback (min(VJ,Hung,Beam)): {self.stats['approx_fallback']} ({100.0*self.stats['approx_fallback']/total:.1f}%)")
        logger.info(f"Available approx methods: {', '.join(self.approx_methods_available) if self.approx_methods_available else 'None'}")


# Test function
def test_gisma_ged_calculator():
    """Test the GED calculator"""
    logger.info("Testing Gisma GED Calculator...")

    # Test using AStar (0.1s timeout) + GREED embedding fallback
    calc = GismaGEDCalculator(
        use_app_for_computation=False,  # Use AStar
        timeout=5  # 5-second timeout
    )

    # Get test graphs
    graph_ids = list(calc.all_graphs.keys())[:10]

    logger.info(f"\nTesting with {len(graph_ids)} graphs...\n")

    for i in range(len(graph_ids)):
        for j in range(i+1, min(i+3, len(graph_ids))):  # Test each graph against the next 2 graphs only
            q = calc.all_graphs[graph_ids[i]]
            g = calc.all_graphs[graph_ids[j]]

            ged = calc.compute_ged(q, g)

            logger.info(f"GED({graph_ids[i]}, {graph_ids[j]}) = {ged:.2f}")
            logger.info(f"  Q: {q.number_of_nodes()} nodes, {q.number_of_edges()} edges")
            logger.info(f"  G: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Print statistics
    logger.info("\n")
    calc.print_stats()


if __name__ == "__main__":
    test_gisma_ged_calculator()
