# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'\nGraphRAG \u589e\u5f3a\u7684\u5de5\u5177\u68c0\u7d22\u7cfb\u7edf\n\n\u57fa\u4e8e\u73b0\u6709\u7684 central.py \u5347\u7ea7\u5230 GraphRAG\uff0c\u589e\u5f3a\u68c0\u7d22\u80fd\u529b\uff1a\n1. \u591a\u8df3\u63a8\u7406\uff1a\u57fa\u4e8e\u4f9d\u8d56\u56fe\u8fdb\u884c\u591a\u8df3\u641c\u7d22\n2. \u793e\u533a\u68c0\u6d4b\uff1a\u5c06\u76f8\u5173\u5de5\u5177\u805a\u7c7b\u6210\u793e\u533a\u5e76\u5efa\u7acb\u5c42\u6b21\u7ed3\u6784\n3. \u5168\u5c40-\u5c40\u90e8\u641c\u7d22\uff1a\u7ed3\u5408\u5168\u5c40\u6458\u8981\u548c\u5c40\u90e8\u7ec6\u8282\u68c0\u7d22\n4. \u5173\u7cfb\u589e\u5f3a\uff1a\u5229\u7528\u5de5\u5177\u95f4\u7684\u8c03\u7528\u5173\u7cfb\u63d0\u5347\u68c0\u7d22\u8d28\u91cf\n5. \u52a8\u6001\u6743\u91cd\uff1a\u6839\u636e\u67e5\u8be2\u7c7b\u578b\u52a8\u6001\u8c03\u6574\u641c\u7d22\u7b56\u7565\n6. \u6267\u884c\u5386\u53f2\uff1a\u8003\u8651\u5de5\u5177\u7684\u5386\u53f2\u6210\u529f\u7387\u548c\u4f7f\u7528\u9891\u7387\n\nAuthor: Based on CentralizedQwenManager\nDate: 2026-01-01\n'

import re
import json
import time
import threading
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from pathlib import Path

# ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

class GraphRAGRetrieval:
    '\n    GraphRAG \u589e\u5f3a\u7684\u5de5\u5177\u68c0\u7d22\u5f15\u64ce\n\n    \u6838\u5fc3\u6539\u8fdb\uff1a\n    - \u5229\u7528\u5de5\u5177\u4f9d\u8d56\u56fe\u8fdb\u884c\u591a\u8df3\u63a8\u7406\n    - \u793e\u533a\u68c0\u6d4b\u548c\u5c42\u6b21\u5316\u7ec4\u7ec7\n    - \u5168\u5c40\u6458\u8981 + \u5c40\u90e8\u8be6\u7ec6\u4fe1\u606f\u7684\u53cc\u5c42\u68c0\u7d22\n    - \u57fa\u4e8e\u6267\u884c\u5386\u53f2\u7684\u52a8\u6001\u6392\u5e8f\n    '

    def __init__(
        self,
        embedding_model_path: str,
        device: str = 'cuda:0',
        enable_community_detection: bool = True,
        enable_history_tracking: bool = True,
        cache_dir: Optional[str] = None
    ):
        "\n        \u521d\u59cb\u5316 GraphRAG \u68c0\u7d22\u5f15\u64ce\n\n        Args:\n            embedding_model_path: SentenceTransformer \u6a21\u578b\u8def\u5f84\n            device: \u8bbe\u5907 ('cuda:0' \u6216 'cpu')\n            enable_community_detection: \u662f\u5426\u542f\u7528\u793e\u533a\u68c0\u6d4b\n            enable_history_tracking: \u662f\u5426\u542f\u7528\u5386\u53f2\u8ddf\u8e2a\n            cache_dir: \u7f13\u5b58\u76ee\u5f55\n        "

        self.lock = threading.RLock()
        self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        self.mcps: Dict[str, Dict[str, Any]] = {}
        self.tool_graph = nx.DiGraph()
        self.tool_embeddings: Dict[str, np.ndarray] = {}
        self.tfidf_matrix = None
        self.tfidf_mcp_ids: List[str] = []

        self.enable_community_detection = enable_community_detection
        self.enable_history_tracking = enable_history_tracking
        self.communities: Dict[str, Set[str]] = {}
        self.community_summaries: Dict[str, str] = {}
        self.global_summary: str = ""

        self.execution_history: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'total_time': 0.0,
            'last_used': 0.0,
            'co_occurrence': Counter()
        })

        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/graphrag_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GraphRAG] Initialized with embedding_model={embedding_model_path}, device={device}")
        print(f"[GraphRAG] Community detection: {enable_community_detection}, History tracking: {enable_history_tracking}")

    def load_tools(self, mcps: Dict[str, Dict[str, Any]], tool_graph: Optional[nx.DiGraph] = None):
        '\n        \u52a0\u8f7d\u5de5\u5177\u6570\u636e\u5e76\u6784\u5efa\u7d22\u5f15\n\n        Args:\n            mcps: \u5de5\u5177\u5143\u6570\u636e\u5b57\u5178\n            tool_graph: \u5de5\u5177\u4f9d\u8d56\u56fe\uff08\u53ef\u9009\uff09\n        '
        with self.lock:
            self.mcps = mcps.copy()

            if tool_graph is not None:
                self.tool_graph = tool_graph.copy()
            else:

                self._build_graph_from_metadata()

            self._build_embeddings()
            self._build_tfidf_index()

            if self.enable_community_detection:
                self._detect_communities()
                self._build_community_summaries()
                self._build_global_summary()

            print(f"[GraphRAG] Loaded {len(self.mcps)} tools, graph has {self.tool_graph.number_of_nodes()} nodes, {self.tool_graph.number_of_edges()} edges")
            if self.enable_community_detection:
                print(f"[GraphRAG] Detected {len(self.communities)} communities")

    def _build_graph_from_metadata(self):
        '\u4ece\u5de5\u5177\u5143\u6570\u636e\u6784\u5efa\u4f9d\u8d56\u56fe'
        self.tool_graph.clear()

        for mcp_name, mcp_data in self.mcps.items():
            step_id = mcp_data.get('step_id', mcp_name)
            self.tool_graph.add_node(step_id, mcp_name=mcp_name)

            requires = mcp_data.get('requires', [])
            if isinstance(requires, str):
                requires = [r.strip() for r in requires.split(',') if r.strip()]

            for req in requires:
                if req in self.mcps:
                    req_step_id = self.mcps[req].get('step_id', req)
                    self.tool_graph.add_edge(req_step_id, step_id)

    def _build_embeddings(self):
        '\u6784\u5efa\u5de5\u5177\u7684\u8bed\u4e49\u5411\u91cf'
        print("[GraphRAG] Building embeddings...")

        for mcp_name, mcp_data in self.mcps.items():

            text_parts = []

            if mcp_data.get('description'):
                text_parts.append(f"Description: {mcp_data['description']}")
            if mcp_data.get('arguments'):
                text_parts.append(f"Arguments: {mcp_data['arguments']}")
            if mcp_data.get('returns'):
                text_parts.append(f"Returns: {mcp_data['returns']}")

            combined_text = " ".join(text_parts) if text_parts else mcp_name
            embedding = self.embedding_model.encode(combined_text, convert_to_numpy=True)
            self.tool_embeddings[mcp_name] = embedding

        print(f"[GraphRAG] Built embeddings for {len(self.tool_embeddings)} tools")

    def _build_tfidf_index(self):
        '\u6784\u5efa TF-IDF \u7d22\u5f15'
        print("[GraphRAG] Building TF-IDF index...")

        texts = []
        self.tfidf_mcp_ids = []

        for mcp_name, mcp_data in self.mcps.items():
            text_parts = []
            if mcp_data.get('description'):
                text_parts.append(mcp_data['description'])
            if mcp_data.get('arguments'):
                text_parts.append(mcp_data['arguments'])
            if mcp_data.get('returns'):
                text_parts.append(mcp_data['returns'])

            combined_text = " ".join(text_parts) if text_parts else mcp_name
            texts.append(combined_text)
            self.tfidf_mcp_ids.append(mcp_name)

        if texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print(f"[GraphRAG] Built TF-IDF index for {len(texts)} tools")
        else:
            self.tfidf_matrix = None
            print("[GraphRAG] Warning: No texts to build TF-IDF index")

    def _detect_communities(self):
        '\n        \u4f7f\u7528\u5c42\u6b21\u805a\u7c7b\u8fdb\u884c\u793e\u533a\u68c0\u6d4b\n\n        \u7ed3\u5408\u56fe\u7ed3\u6784\u548c\u8bed\u4e49\u76f8\u4f3c\u5ea6\u8fdb\u884c\u805a\u7c7b\n        '
        if not self.tool_embeddings:
            print("[GraphRAG] No embeddings available for community detection")
            return

        print("[GraphRAG] Detecting communities...")

        mcp_names = list(self.tool_embeddings.keys())
        n_tools = len(mcp_names)

        if n_tools < 2:

            self.communities = {'community_0': set(mcp_names)}
            return

        embeddings = np.array([self.tool_embeddings[name] for name in mcp_names])
        semantic_sim = cosine_similarity(embeddings)

        structural_sim = np.zeros((n_tools, n_tools))
        for i, name_i in enumerate(mcp_names):
            for j, name_j in enumerate(mcp_names):
                if i == j:
                    structural_sim[i, j] = 1.0
                else:

                    step_i = self.mcps[name_i].get('step_id', name_i)
                    step_j = self.mcps[name_j].get('step_id', name_j)

                    if self.tool_graph.has_node(step_i) and self.tool_graph.has_node(step_j):
                        neighbors_i = set(self.tool_graph.predecessors(step_i)) | set(self.tool_graph.successors(step_i))
                        neighbors_j = set(self.tool_graph.predecessors(step_j)) | set(self.tool_graph.successors(step_j))
                        common = len(neighbors_i & neighbors_j)
                        total = len(neighbors_i | neighbors_j)
                        structural_sim[i, j] = common / (total + 1e-6)

        combined_sim = 0.7 * semantic_sim + 0.3 * structural_sim

        distance = 1 - combined_sim

        n_clusters = min(max(2, n_tools // 5), 10)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance)

        self.communities = {}
        for i, label in enumerate(labels):
            community_id = f'community_{label}'
            if community_id not in self.communities:
                self.communities[community_id] = set()
            self.communities[community_id].add(mcp_names[i])

        print(f"[GraphRAG] Detected {len(self.communities)} communities:")
        for comm_id, members in self.communities.items():
            print(f"  - {comm_id}: {len(members)} tools")

    def _build_community_summaries(self):
        '\u4e3a\u6bcf\u4e2a\u793e\u533a\u6784\u5efa\u6458\u8981'
        print("[GraphRAG] Building community summaries...")

        for comm_id, members in self.communities.items():

            descriptions = []
            tool_names = []

            for mcp_name in members:
                mcp_data = self.mcps.get(mcp_name, {})
                if mcp_data.get('description'):
                    descriptions.append(mcp_data['description'])
                tool_names.append(mcp_name)

            if descriptions:

                all_text = " ".join(descriptions)
                words = re.findall(r'\b\w+\b', all_text.lower())
                word_counts = Counter(words)

                stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'])
                keywords = [word for word, count in word_counts.most_common(10) if word not in stopwords and len(word) > 3]

                summary = f"Community {comm_id} contains {len(members)} tools related to: {', '.join(keywords[:5])}. "
                summary += f"Tools: {', '.join(list(tool_names)[:10])}"
                if len(tool_names) > 10:
                    summary += f" and {len(tool_names) - 10} more"
            else:
                summary = f"Community {comm_id} contains {len(members)} tools: {', '.join(list(tool_names)[:10])}"

            self.community_summaries[comm_id] = summary

        print(f"[GraphRAG] Built summaries for {len(self.community_summaries)} communities")

    def _build_global_summary(self):
        '\u6784\u5efa\u5168\u5c40\u6458\u8981'
        community_texts = []
        for comm_id in sorted(self.community_summaries.keys()):
            community_texts.append(self.community_summaries[comm_id])

        self.global_summary = f"Total {len(self.mcps)} tools organized into {len(self.communities)} communities. " + " ".join(community_texts)
        print(f"[GraphRAG] Built global summary ({len(self.global_summary)} chars)")

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '\n        \u8bed\u4e49\u641c\u7d22\uff08\u5411\u540e\u517c\u5bb9\uff09\n\n        Args:\n            query: \u67e5\u8be2\u6587\u672c\n            top_k: \u8fd4\u56de\u7ed3\u679c\u6570\u91cf\n\n        Returns:\n            \u5339\u914d\u7684\u5de5\u5177\u5217\u8868\uff0c\u6bcf\u4e2a\u5305\u542b mcp_id, score, content\n        '
        if not self.tool_embeddings:
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        scores = []
        for mcp_name, embedding in self.tool_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            scores.append((mcp_name, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mcp_name, score in scores[:top_k]:
            results.append({
                'mcp_id': mcp_name,
                'score': float(score),
                'field': 'description',
                'content': self.mcps[mcp_name].get('description', '')
            })

        return results

    def text_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        '\n        TF-IDF \u6587\u672c\u641c\u7d22\uff08\u5411\u540e\u517c\u5bb9\uff09\n\n        Args:\n            query: \u67e5\u8be2\u6587\u672c\n            top_k: \u8fd4\u56de\u7ed3\u679c\u6570\u91cf\n\n        Returns:\n            \u5339\u914d\u7684\u5de5\u5177\u5217\u8868\n        '
        if self.tfidf_matrix is None or not self.tfidf_mcp_ids:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        actual_top_k = min(top_k, len(similarities))
        top_indices = similarities.argsort()[-actual_top_k:][::-1]

        results = []
        for idx in top_indices:
            mcp_id = self.tfidf_mcp_ids[idx]
            if mcp_id in self.mcps:
                results.append({
                    'mcp_id': mcp_id,
                    'score': float(similarities[idx]),
                    'content': self.mcps[mcp_id]
                })

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        score_threshold: float = 0.2
    ) -> List[List[Dict[str, Any]]]:
        "\n        \u6df7\u5408\u641c\u7d22\uff08\u5411\u540e\u517c\u5bb9\u63a5\u53e3\uff09\n\n        Args:\n            query: \u67e5\u8be2\u6587\u672c\n            top_k: \u8fd4\u56de\u7ed3\u679c\u6570\u91cf\n            weights: \u6743\u91cd\u5b57\u5178 {'text': w1, 'semantic': w2}\n            score_threshold: \u5206\u6570\u9608\u503c\n\n        Returns:\n            \u5339\u914d\u7684\u5de5\u5177\u5217\u8868\uff08\u6bcf\u4e2a\u5de5\u5177\u5305\u88c5\u5728\u5217\u8868\u4e2d\u4ee5\u517c\u5bb9\u65e7\u63a5\u53e3\uff09\n        "
        weights = weights or {'text': 0.5, 'semantic': 0.5}
        score_board = defaultdict(float)

        text_results = self.text_search(query, top_k * 2)
        for r in text_results:
            score_board[r['mcp_id']] += weights.get('text', 0.5) * r['score']

        semantic_results = self.semantic_search(query, top_k * 2)
        for r in semantic_results:
            score_board[r['mcp_id']] += weights.get('semantic', 0.5) * r['score']

        filtered_results = [(mcp_id, score) for mcp_id, score in score_board.items() if score > score_threshold]
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]

        detailed_results = []
        for mcp_id, score in sorted_results:
            mcp_item = self.mcps.get(mcp_id)
            if mcp_item:
                mcp_item_with_score = mcp_item.copy()
                mcp_item_with_score['search_score'] = score
                detailed_results.append([mcp_item_with_score])

        return detailed_results

    def graph_enhanced_search(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
        use_community: bool = True,
        use_history: bool = True,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        '\n        GraphRAG \u589e\u5f3a\u641c\u7d22\uff08\u6838\u5fc3\u65b9\u6cd5\uff09\n\n        Args:\n            query: \u67e5\u8be2\u6587\u672c\n            top_k: \u8fd4\u56de\u7ed3\u679c\u6570\u91cf\n            max_hops: \u6700\u5927\u8df3\u6570\uff08\u591a\u8df3\u63a8\u7406\uff09\n            use_community: \u662f\u5426\u4f7f\u7528\u793e\u533a\u4fe1\u606f\n            use_history: \u662f\u5426\u4f7f\u7528\u6267\u884c\u5386\u53f2\n            weights: \u6743\u91cd\u914d\u7f6e\n\n        Returns:\n            \u589e\u5f3a\u7684\u641c\u7d22\u7ed3\u679c\u5217\u8868\n        '
        weights = weights or {
            'semantic': 0.3,
            'text': 0.2,
            'graph': 0.2,
            'community': 0.15,
            'history': 0.15
        }

        print(f"[GraphRAG] Graph-enhanced search: query='{query[:50]}...', top_k={top_k}, max_hops={max_hops}")

        semantic_results = self.semantic_search(query, top_k * 3)
        text_results = self.text_search(query, top_k * 3)

        candidates = set()
        base_scores = defaultdict(float)

        for r in semantic_results:
            mcp_id = r['mcp_id']
            candidates.add(mcp_id)
            base_scores[mcp_id] += weights['semantic'] * r['score']

        for r in text_results:
            mcp_id = r['mcp_id']
            candidates.add(mcp_id)
            base_scores[mcp_id] += weights['text'] * r['score']

        graph_scores = self._compute_graph_scores(candidates, max_hops=max_hops)

        community_scores = {}
        if use_community and self.communities:
            community_scores = self._compute_community_scores(query, candidates)

        history_scores = {}
        if use_history and self.enable_history_tracking:
            history_scores = self._compute_history_scores(candidates)

        final_scores = {}
        all_candidates = candidates | set(graph_scores.keys())

        for mcp_id in all_candidates:
            score = 0.0
            score += base_scores.get(mcp_id, 0.0)
            score += weights.get('graph', 0.2) * graph_scores.get(mcp_id, 0.0)
            score += weights.get('community', 0.15) * community_scores.get(mcp_id, 0.0)
            score += weights.get('history', 0.15) * history_scores.get(mcp_id, 0.0)
            final_scores[mcp_id] = score

        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        detailed_results = []
        for mcp_id, score in sorted_results:
            mcp_item = self.mcps.get(mcp_id)
            if mcp_item:
                result = mcp_item.copy()
                result['search_score'] = float(score)
                result['mcp_id'] = mcp_id
                result['mcp_name'] = mcp_id

                result['score_breakdown'] = {
                    'base': base_scores.get(mcp_id, 0.0),
                    'graph': graph_scores.get(mcp_id, 0.0),
                    'community': community_scores.get(mcp_id, 0.0),
                    'history': history_scores.get(mcp_id, 0.0)
                }

                detailed_results.append(result)

        print(f"[GraphRAG] Returned {len(detailed_results)} results")
        return detailed_results

    def _compute_graph_scores(self, seed_tools: Set[str], max_hops: int = 2) -> Dict[str, float]:
        '\n        \u8ba1\u7b97\u56fe\u7ed3\u6784\u5206\u6570\uff08\u591a\u8df3\u63a8\u7406\uff09\n\n        \u4ece\u79cd\u5b50\u5de5\u5177\u51fa\u53d1\uff0c\u901a\u8fc7\u4f9d\u8d56\u56fe\u6269\u5c55\uff0c\u8ddd\u79bb\u8d8a\u8fd1\u5206\u6570\u8d8a\u9ad8\n        '
        graph_scores = {}

        for seed in seed_tools:
            step_id = self.mcps[seed].get('step_id', seed)
            if not self.tool_graph.has_node(step_id):
                continue

            visited = {step_id: 0}
            queue = [(step_id, 0)]

            while queue:
                current, distance = queue.pop(0)
                if distance >= max_hops:
                    continue

                for pred in self.tool_graph.predecessors(current):
                    if pred not in visited:
                        visited[pred] = distance + 1
                        queue.append((pred, distance + 1))

                for succ in self.tool_graph.successors(current):
                    if succ not in visited:
                        visited[succ] = distance + 1
                        queue.append((succ, distance + 1))

            for node, dist in visited.items():

                mcp_name = self.tool_graph.nodes[node].get('mcp_name', None)
                if mcp_name and mcp_name in self.mcps:
                    decay_factor = 0.5 ** dist
                    graph_scores[mcp_name] = max(graph_scores.get(mcp_name, 0.0), decay_factor)

        return graph_scores

    def _compute_community_scores(self, query: str, candidates: Set[str]) -> Dict[str, float]:
        '\n        \u8ba1\u7b97\u793e\u533a\u76f8\u5173\u6027\u5206\u6570\n\n        \u5982\u679c\u67e5\u8be2\u4e0e\u67d0\u4e2a\u793e\u533a\u7684\u6458\u8981\u76f8\u4f3c\uff0c\u5219\u8be5\u793e\u533a\u5185\u7684\u5de5\u5177\u83b7\u5f97\u52a0\u5206\n        '
        if not self.community_summaries:
            return {}

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        community_scores = {}

        for comm_id, summary in self.community_summaries.items():
            summary_embedding = self.embedding_model.encode(summary, convert_to_numpy=True)
            similarity = cosine_similarity([query_embedding], [summary_embedding])[0][0]

            for mcp_name in self.communities[comm_id]:
                if mcp_name in candidates:
                    community_scores[mcp_name] = max(community_scores.get(mcp_name, 0.0), similarity)

        return community_scores

    def _compute_history_scores(self, candidates: Set[str]) -> Dict[str, float]:
        '\n        \u8ba1\u7b97\u57fa\u4e8e\u6267\u884c\u5386\u53f2\u7684\u5206\u6570\n\n        \u8003\u8651\u6210\u529f\u7387\u3001\u4f7f\u7528\u9891\u7387\u3001\u6700\u8fd1\u4f7f\u7528\u65f6\u95f4\u7b49\u56e0\u7d20\n        '
        history_scores = {}
        current_time = time.time()

        for mcp_name in candidates:
            if mcp_name not in self.execution_history:
                history_scores[mcp_name] = 0.0
                continue

            hist = self.execution_history[mcp_name]
            total_exec = hist['success_count'] + hist['failure_count']

            if total_exec == 0:
                history_scores[mcp_name] = 0.0
                continue

            success_rate = hist['success_count'] / total_exec

            frequency_score = min(1.0, total_exec / 100.0)

            time_diff = current_time - hist['last_used']
            recency_score = np.exp(-time_diff / 86400)

            score = 0.5 * success_rate + 0.3 * frequency_score + 0.2 * recency_score
            history_scores[mcp_name] = score

        return history_scores

    def record_execution(
        self,
        mcp_name: str,
        success: bool,
        execution_time: float = 0.0,
        co_used_tools: Optional[List[str]] = None
    ):
        '\n        \u8bb0\u5f55\u5de5\u5177\u6267\u884c\u5386\u53f2\n\n        Args:\n            mcp_name: \u5de5\u5177\u540d\u79f0\n            success: \u662f\u5426\u6210\u529f\n            execution_time: \u6267\u884c\u65f6\u95f4\uff08\u79d2\uff09\n            co_used_tools: \u540c\u65f6\u4f7f\u7528\u7684\u5176\u4ed6\u5de5\u5177\n        '
        if not self.enable_history_tracking:
            return

        with self.lock:
            hist = self.execution_history[mcp_name]

            if success:
                hist['success_count'] += 1
            else:
                hist['failure_count'] += 1

            hist['total_time'] += execution_time
            hist['last_used'] = time.time()

            if co_used_tools:
                for other_tool in co_used_tools:
                    if other_tool != mcp_name:
                        hist['co_occurrence'][other_tool] += 1

    def get_execution_stats(self, mcp_name: str) -> Dict[str, Any]:
        '\u83b7\u53d6\u5de5\u5177\u7684\u6267\u884c\u7edf\u8ba1\u4fe1\u606f'
        if mcp_name not in self.execution_history:
            return {}

        hist = self.execution_history[mcp_name]
        total_exec = hist['success_count'] + hist['failure_count']

        return {
            'success_count': hist['success_count'],
            'failure_count': hist['failure_count'],
            'success_rate': hist['success_count'] / total_exec if total_exec > 0 else 0.0,
            'total_executions': total_exec,
            'avg_time': hist['total_time'] / total_exec if total_exec > 0 else 0.0,
            'last_used': hist['last_used'],
            'frequently_co_used': hist['co_occurrence'].most_common(5)
        }

    def update_tool(self, mcp_name: str, mcp_data: Dict[str, Any]):
        '\u66f4\u65b0\u5355\u4e2a\u5de5\u5177'
        with self.lock:
            self.mcps[mcp_name] = mcp_data

            text_parts = []
            if mcp_data.get('description'):
                text_parts.append(f"Description: {mcp_data['description']}")
            if mcp_data.get('arguments'):
                text_parts.append(f"Arguments: {mcp_data['arguments']}")
            if mcp_data.get('returns'):
                text_parts.append(f"Returns: {mcp_data['returns']}")

            combined_text = " ".join(text_parts) if text_parts else mcp_name
            self.tool_embeddings[mcp_name] = self.embedding_model.encode(combined_text, convert_to_numpy=True)

            self._build_tfidf_index()

            print(f"[GraphRAG] Updated tool: {mcp_name}")

    def remove_tool(self, mcp_name: str):
        '\u5220\u9664\u5de5\u5177'
        with self.lock:
            if mcp_name in self.mcps:
                del self.mcps[mcp_name]
            if mcp_name in self.tool_embeddings:
                del self.tool_embeddings[mcp_name]
            if mcp_name in self.execution_history:
                del self.execution_history[mcp_name]

            step_id = None
            for node, data in self.tool_graph.nodes(data=True):
                if data.get('mcp_name') == mcp_name:
                    step_id = node
                    break
            if step_id is not None:
                self.tool_graph.remove_node(step_id)

            self._build_tfidf_index()

            print(f"[GraphRAG] Removed tool: {mcp_name}")

    def save_cache(self, cache_path: Optional[str] = None):
        '\u4fdd\u5b58\u7f13\u5b58\u5230\u78c1\u76d8'
        if cache_path is None:
            cache_path = self.cache_dir / "graphrag_cache.json"

        cache_data = {
            'execution_history': dict(self.execution_history),
            'communities': {k: list(v) for k, v in self.communities.items()},
            'community_summaries': self.community_summaries,
            'global_summary': self.global_summary
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print(f"[GraphRAG] Cache saved to {cache_path}")

    def load_cache(self, cache_path: Optional[str] = None):
        '\u4ece\u78c1\u76d8\u52a0\u8f7d\u7f13\u5b58'
        if cache_path is None:
            cache_path = self.cache_dir / "graphrag_cache.json"

        if not Path(cache_path).exists():
            print(f"[GraphRAG] Cache file not found: {cache_path}")
            return

        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        for mcp_name, hist in cache_data.get('execution_history', {}).items():
            self.execution_history[mcp_name] = defaultdict(lambda: 0)
            self.execution_history[mcp_name].update(hist)

            if 'co_occurrence' in hist:
                self.execution_history[mcp_name]['co_occurrence'] = Counter(hist['co_occurrence'])

        self.communities = {k: set(v) for k, v in cache_data.get('communities', {}).items()}
        self.community_summaries = cache_data.get('community_summaries', {})
        self.global_summary = cache_data.get('global_summary', '')

        print(f"[GraphRAG] Cache loaded from {cache_path}")

def create_graphrag_retrieval(
    mcps: Dict[str, Dict[str, Any]],
    tool_graph: Optional[nx.DiGraph] = None,
    embedding_model_path: str = 'sentence-transformers/all-MiniLM-L6-v2',
    device: str = 'cuda:0',
    enable_community_detection: bool = True,
    enable_history_tracking: bool = True,
    cache_dir: Optional[str] = None
) -> GraphRAGRetrieval:
    '\n    \u4fbf\u6377\u51fd\u6570\uff1a\u521b\u5efa\u5e76\u521d\u59cb\u5316 GraphRAG \u68c0\u7d22\u5f15\u64ce\n\n    Args:\n        mcps: \u5de5\u5177\u5143\u6570\u636e\u5b57\u5178\n        tool_graph: \u5de5\u5177\u4f9d\u8d56\u56fe\uff08\u53ef\u9009\uff09\n        embedding_model_path: SentenceTransformer \u6a21\u578b\u8def\u5f84\n        device: \u8bbe\u5907\n        enable_community_detection: \u542f\u7528\u793e\u533a\u68c0\u6d4b\n        enable_history_tracking: \u542f\u7528\u5386\u53f2\u8ddf\u8e2a\n        cache_dir: \u7f13\u5b58\u76ee\u5f55\n\n    Returns:\n        \u521d\u59cb\u5316\u597d\u7684 GraphRAGRetrieval \u5b9e\u4f8b\n    '
    retrieval = GraphRAGRetrieval(
        embedding_model_path=embedding_model_path,
        device=device,
        enable_community_detection=enable_community_detection,
        enable_history_tracking=enable_history_tracking,
        cache_dir=cache_dir
    )

    retrieval.load_tools(mcps, tool_graph)

    return retrieval

if __name__ == "__main__":

    print("=" * 60)
    print('GraphRAG \u68c0\u7d22\u5f15\u64ce\u6d4b\u8bd5')
    print("=" * 60)

    sample_mcps = {
        'web_search': {
            'mcp_name': 'web_search',
            'description': 'Search the web for information using Bing API',
            'arguments': 'query: str - search query',
            'returns': 'List of search results',
            'requires': [],
            'code': 'web_search(query)',
            'step_id': 1
        },
        'extract_content': {
            'mcp_name': 'extract_content',
            'description': 'Extract main content from a web page',
            'arguments': 'url: str - web page URL',
            'returns': 'Extracted text content',
            'requires': ['web_search'],
            'code': 'extract_content(url)',
            'step_id': 2
        },
        'summarize_text': {
            'mcp_name': 'summarize_text',
            'description': 'Summarize long text into a concise summary',
            'arguments': 'text: str - text to summarize',
            'returns': 'Summary string',
            'requires': ['extract_content'],
            'code': 'summarize_text(text)',
            'step_id': 3
        },
        'python_executor': {
            'mcp_name': 'python_executor',
            'description': 'Execute Python code in a sandbox environment',
            'arguments': 'code: str - Python code to execute',
            'returns': 'Execution result',
            'requires': [],
            'code': 'python_executor(code)',
            'step_id': 4
        }
    }

    print('\n1. \u521b\u5efa GraphRAG \u68c0\u7d22\u5f15\u64ce...')
    retrieval = create_graphrag_retrieval(
        mcps=sample_mcps,
        enable_community_detection=True,
        enable_history_tracking=True,
        device='cpu'
    )

    print('\n2. \u6d4b\u8bd5\u8bed\u4e49\u641c\u7d22...')
    results = retrieval.semantic_search("search for information", top_k=2)
    for r in results:
        print(f"  - {r['mcp_id']}: score={r['score']:.4f}")

    print('\n3. \u6d4b\u8bd5\u6df7\u5408\u641c\u7d22...')
    results = retrieval.hybrid_search("find and summarize web content", top_k=3)
    for r_list in results:
        for r in r_list:
            print(f"  - {r.get('mcp_name')}: score={r.get('search_score', 0):.4f}")

    print('\n4. \u6d4b\u8bd5 GraphRAG \u589e\u5f3a\u641c\u7d22...')
    results = retrieval.graph_enhanced_search("search and extract information from websites", top_k=3, max_hops=2)
    for r in results:
        print(f"  - {r['mcp_name']}: total_score={r['search_score']:.4f}")
        if 'score_breakdown' in r:
            print(f"    breakdown: {r['score_breakdown']}")

    print('\n5. \u6d4b\u8bd5\u6267\u884c\u5386\u53f2\u8bb0\u5f55...')
    retrieval.record_execution('web_search', success=True, execution_time=1.5)
    retrieval.record_execution('web_search', success=True, execution_time=1.2)
    retrieval.record_execution('extract_content', success=False, execution_time=0.5)

    stats = retrieval.get_execution_stats('web_search')
    print(f"  web_search stats: {stats}")

    print("\n" + "=" * 60)
    print('\u6d4b\u8bd5\u5b8c\u6210\uff01')
    print("=" * 60)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
