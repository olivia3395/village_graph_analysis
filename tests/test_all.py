"""
tests/test_all.py — Unit tests for Village Graph Analysis.

Run:
    python -m pytest tests/ -v
    python tests/test_all.py

Groups:
  A — Config
  B — Synthetic data generation
  C — Preprocessing
  D — Graph builder
  E — Graph features (Laplacian, adjacency)
  F — Spectral clustering
  G — Community detection
  H — Node2Vec embedding
  I — Typology labelling
  J — Development scoring
  K — Evaluation metrics
  L — End-to-end pipeline smoke test
"""

import sys, os, math, unittest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


# =============================================================================
# A — Config
# =============================================================================
class TestConfig(unittest.TestCase):
    def test_default_config(self):
        from config import Config
        cfg = Config()
        self.assertEqual(cfg.graph.edge_strategy, "hybrid")
        self.assertEqual(cfg.clustering.n_clusters, 5)
        self.assertAlmostEqual(
            cfg.graph.spatial_weight + cfg.graph.feature_weight, 1.0, places=5
        )

    def test_feature_groups_defined(self):
        from config import Config
        cfg = Config()
        self.assertIn("economic", cfg.data.feature_groups)
        self.assertIn("geographic", cfg.data.feature_groups)
        self.assertIn("infrastructure", cfg.data.feature_groups)

    def test_summary_runs(self):
        from config import Config
        s = Config().summary()
        self.assertIn("GRAPH", s)
        self.assertIn("CLUSTERING", s)


# =============================================================================
# B — Synthetic data generation
# =============================================================================
class TestSynthetic(unittest.TestCase):
    def setUp(self):
        from data.synthetic import generate_village_data
        self.df, self.labels = generate_village_data(n_villages=50, seed=0)

    def test_shape(self):
        self.assertEqual(len(self.df), 50)
        self.assertGreater(len(self.df.columns), 10)

    def test_labels_range(self):
        self.assertTrue(set(self.labels.tolist()).issubset({0, 1, 2, 3, 4}))

    def test_all_five_types_present(self):
        self.assertEqual(len(set(self.labels.tolist())), 5)

    def test_coordinates_in_ji_county_range(self):
        self.assertTrue((self.df["longitude"] > 116).all())
        self.assertTrue((self.df["longitude"] < 119).all())
        self.assertTrue((self.df["latitude"] > 39).all())
        self.assertTrue((self.df["latitude"] < 41).all())

    def test_no_nan(self):
        self.assertFalse(self.df.isnull().any().any())

    def test_income_positive(self):
        self.assertTrue((self.df["income_per_capita_cny"] > 0).all())

    def test_ratios_in_unit_interval(self):
        for col in ("employment_rate", "road_quality", "education_index"):
            self.assertTrue((self.df[col] >= 0).all(), f"{col} below 0")
            self.assertTrue((self.df[col] <= 1).all(), f"{col} above 1")


# =============================================================================
# C — Preprocessing
# =============================================================================
class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        from data.synthetic import generate_village_data
        from config import Config
        self.df, _ = generate_village_data(n_villages=40, seed=1)
        self.cfg   = Config()

    def test_impute_missing(self):
        from data.preprocessing import impute_missing
        df_miss = self.df.copy()
        df_miss.iloc[0, 0] = float("nan")
        df_out = impute_missing(df_miss)
        self.assertFalse(df_out.isnull().any().any())

    def test_minmax_range(self):
        from data.preprocessing import minmax_normalise
        df_norm = minmax_normalise(self.df, ["income_per_capita_cny", "population"])
        self.assertTrue((df_norm["income_per_capita_cny"] >= 0).all())
        self.assertTrue((df_norm["income_per_capita_cny"] <= 1).all())

    def test_composite_indicators_created(self):
        from data.preprocessing import compute_composite_indicators
        df_out = compute_composite_indicators(self.df)
        for col in ("economic_vitality", "development_index", "revitalization_need"):
            self.assertIn(col, df_out.columns)

    def test_revitalization_need_inverse_of_development(self):
        from data.preprocessing import compute_composite_indicators
        df_out = compute_composite_indicators(self.df)
        diff   = (df_out["development_index"] + df_out["revitalization_need"] - 1.0).abs()
        self.assertTrue((diff < 1e-6).all())

    def test_full_pipeline_shape(self):
        from data.preprocessing import preprocess
        _, X, feat_names = preprocess(self.df, self.cfg.data.feature_groups)
        self.assertEqual(X.shape[0], 40)
        self.assertGreater(X.shape[1], 5)
        self.assertEqual(len(feat_names), X.shape[1])
        self.assertFalse(np.isnan(X).any())


# =============================================================================
# D — Graph builder
# =============================================================================
class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        from data.synthetic import generate_village_data
        from data.preprocessing import preprocess
        from graph.builder import VillageGraphBuilder
        from config import Config
        cfg = Config()
        cfg.graph.edge_strategy = "hybrid"
        cfg.graph.distance_threshold_km = 20.0
        self.df, self.labels = generate_village_data(n_villages=30, seed=2)
        _, self.X, _ = preprocess(self.df, cfg.data.feature_groups)
        self.builder = VillageGraphBuilder(cfg.graph)
        self.G = self.builder.build(self.df, self.X)

    def test_node_count(self):
        import networkx as nx
        self.assertEqual(self.G.number_of_nodes(), 30)

    def test_has_edges(self):
        self.assertGreater(self.G.number_of_edges(), 0)

    def test_edge_weights_positive(self):
        import networkx as nx
        for u, v, d in self.G.edges(data=True):
            self.assertGreater(d["weight"], 0)
            self.assertLessEqual(d["weight"], 1.0)

    def test_node_attributes(self):
        node = list(self.G.nodes(data=True))[0]
        attrs = node[1]
        self.assertIn("lon", attrs)
        self.assertIn("lat", attrs)

    def test_graph_attribute_W(self):
        self.assertIn("W", self.G.graph)
        W = self.G.graph["W"]
        self.assertEqual(W.shape, (30, 30))


# =============================================================================
# E — Graph features (Laplacian, adjacency)
# =============================================================================
class TestGraphFeatures(unittest.TestCase):
    def setUp(self):
        from data.synthetic import generate_village_data
        from data.preprocessing import preprocess
        from graph.builder import VillageGraphBuilder
        from config import Config
        cfg = Config()
        self.df, _ = generate_village_data(n_villages=20, seed=3)
        _, self.X, _ = preprocess(self.df, cfg.data.feature_groups)
        self.G = VillageGraphBuilder(cfg.graph).build(self.df, self.X)
        self.village_ids = list(self.df.index)

    def test_adjacency_matrix_shape(self):
        from graph.features import get_adjacency_matrix
        A = get_adjacency_matrix(self.G, self.village_ids)
        self.assertEqual(A.shape, (20, 20))

    def test_adjacency_matrix_symmetric(self):
        from graph.features import get_adjacency_matrix
        A = get_adjacency_matrix(self.G, self.village_ids)
        self.assertTrue(np.allclose(A, A.T, atol=1e-6))

    def test_laplacian_psd(self):
        from graph.features import get_adjacency_matrix, get_laplacian
        A = get_adjacency_matrix(self.G, self.village_ids)
        L = get_laplacian(A, normalised=True)
        eigvals = np.linalg.eigvalsh(L)
        self.assertTrue((eigvals >= -1e-6).all(), "Laplacian not PSD")

    def test_graph_structural_features(self):
        from graph.features import extract_graph_features
        gf = extract_graph_features(self.G)
        self.assertEqual(len(gf), 20)
        for col in ("g_degree", "g_clustering", "g_pagerank"):
            self.assertIn(col, gf.columns)
        self.assertTrue((gf["g_degree"] >= 0).all())


# =============================================================================
# F — Spectral clustering
# =============================================================================
class TestSpectralClustering(unittest.TestCase):
    def _build(self, n=50):
        from data.synthetic import generate_village_data
        from data.preprocessing import preprocess
        from graph.builder import VillageGraphBuilder
        from config import Config
        cfg = Config()
        cfg.clustering.n_clusters = 5
        df, labels_true = generate_village_data(n_villages=n, seed=4)
        _, X, _ = preprocess(df, cfg.data.feature_groups)
        G = VillageGraphBuilder(cfg.graph).build(df, X)
        return G, list(df.index), labels_true, cfg

    def test_label_count(self):
        from models.spectral import SpectralClusterer
        G, vids, _, cfg = self._build()
        sc = SpectralClusterer(cfg.clustering)
        labels = sc.fit_predict(G, vids)
        self.assertEqual(len(labels), len(vids))

    def test_labels_in_range(self):
        from models.spectral import SpectralClusterer
        G, vids, _, cfg = self._build()
        sc = SpectralClusterer(cfg.clustering)
        labels = sc.fit_predict(G, vids)
        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels < cfg.clustering.n_clusters).all())

    def test_embedding_shape(self):
        from models.spectral import SpectralClusterer
        G, vids, _, cfg = self._build()
        sc = SpectralClusterer(cfg.clustering)
        sc.fit_predict(G, vids)
        self.assertEqual(sc.embedding.shape[0], len(vids))
        self.assertEqual(sc.embedding.shape[1], cfg.clustering.n_clusters)

    def test_fiedler_vector_length(self):
        from models.spectral import SpectralClusterer
        G, vids, _, cfg = self._build()
        sc = SpectralClusterer(cfg.clustering)
        sc.fit_predict(G, vids)
        fv = sc.fiedler_vector()
        self.assertEqual(len(fv), len(vids))


# =============================================================================
# G — Community detection
# =============================================================================
class TestCommunityDetection(unittest.TestCase):
    def _build(self, n=40):
        from data.synthetic import generate_village_data
        from data.preprocessing import preprocess
        from graph.builder import VillageGraphBuilder
        from config import Config
        cfg = Config()
        df, _ = generate_village_data(n_villages=n, seed=5)
        _, X, _ = preprocess(df, cfg.data.feature_groups)
        G = VillageGraphBuilder(cfg.graph).build(df, X)
        return G, list(df.index), cfg

    def test_louvain_returns_labels(self):
        from models.community import LouvainDetector
        G, vids, cfg = self._build()
        det = LouvainDetector(cfg.clustering)
        labels = det.fit_predict(G, vids)
        self.assertEqual(len(labels), len(vids))
        self.assertGreater(len(set(labels.tolist())), 0)

    def test_girvan_newman_k_communities(self):
        from models.community import GirvanNewmanDetector
        G, vids, cfg = self._build(n=20)
        cfg.clustering.n_clusters = 3
        det = GirvanNewmanDetector(cfg.clustering)
        labels = det.fit_predict(G, vids)
        self.assertEqual(len(labels), len(vids))
        self.assertLessEqual(len(set(labels.tolist())), cfg.clustering.n_clusters + 1)


# =============================================================================
# H — Node2Vec embedding
# =============================================================================
class TestNode2Vec(unittest.TestCase):
    def _build(self, n=30):
        from data.synthetic import generate_village_data
        from data.preprocessing import preprocess
        from graph.builder import VillageGraphBuilder
        from config import Config
        cfg = Config()
        cfg.clustering.n2v_num_walks   = 5
        cfg.clustering.n2v_walk_length = 10
        df, _ = generate_village_data(n_villages=n, seed=6)
        _, X, _ = preprocess(df, cfg.data.feature_groups)
        G = VillageGraphBuilder(cfg.graph).build(df, X)
        return G, list(df.index), cfg

    def test_embedding_shape(self):
        from models.graph_embedding import Node2VecEmbedder
        G, vids, cfg = self._build()
        n2v = Node2VecEmbedder(cfg.clustering)
        n2v.fit(G, vids)
        self.assertEqual(n2v.embedding.shape[0], len(vids))

    def test_predict_labels(self):
        from models.graph_embedding import Node2VecEmbedder
        G, vids, cfg = self._build()
        n2v = Node2VecEmbedder(cfg.clustering)
        labels = n2v.fit_predict(G, vids)
        self.assertEqual(len(labels), len(vids))
        self.assertGreater(len(set(labels.tolist())), 1)


# =============================================================================
# I — Typology labelling
# =============================================================================
class TestTypology(unittest.TestCase):
    def setUp(self):
        from data.synthetic import generate_village_data
        from data.preprocessing import preprocess, compute_composite_indicators
        from config import Config
        cfg = Config()
        self.df, self.true_labels = generate_village_data(n_villages=100, seed=7)
        self.df = compute_composite_indicators(self.df)
        self.df, self.X, _ = preprocess(self.df, cfg.data.feature_groups)
        self.cfg = cfg

    def test_label_clusters_returns_dict(self):
        from analysis.typology import label_clusters
        labels = np.arange(len(self.df)) % 5
        mapping = label_clusters(self.df, labels, list(self.df.index), self.cfg.typology)
        self.assertIsInstance(mapping, dict)
        self.assertEqual(len(mapping), 5)

    def test_typology_names_are_strings(self):
        from analysis.typology import label_clusters
        labels = np.arange(len(self.df)) % 5
        mapping = label_clusters(self.df, labels, list(self.df.index), self.cfg.typology)
        for v in mapping.values():
            self.assertIsInstance(v, str)

    def test_typology_profile_shape(self):
        from analysis.typology import label_clusters, typology_profile
        labels = np.arange(len(self.df)) % 5
        mapping = label_clusters(self.df, labels, list(self.df.index), self.cfg.typology)
        profile = typology_profile(self.df, labels, mapping)
        self.assertEqual(len(profile), 5)
        self.assertGreater(len(profile.columns), 0)


# =============================================================================
# J — Development scoring
# =============================================================================
class TestDevelopment(unittest.TestCase):
    def setUp(self):
        from data.synthetic import generate_village_data
        from data.preprocessing import preprocess
        from config import Config
        cfg = Config()
        self.df, _ = generate_village_data(n_villages=50, seed=8)
        self.df, _, _ = preprocess(self.df, cfg.data.feature_groups)

    def test_scores_added(self):
        from analysis.development import compute_development_scores
        df_out = compute_development_scores(self.df)
        for col in ("development_index", "revitalization_need",
                    "growth_potential", "priority_tier"):
            self.assertIn(col, df_out.columns)

    def test_scores_in_range(self):
        from analysis.development import compute_development_scores
        df_out = compute_development_scores(self.df)
        for col in ("development_index", "revitalization_need", "growth_potential"):
            self.assertTrue((df_out[col] >= 0).all(), f"{col} below 0")
            self.assertTrue((df_out[col] <= 1).all(), f"{col} above 1")

    def test_priority_tiers(self):
        from analysis.development import compute_development_scores
        df_out = compute_development_scores(self.df)
        valid  = {"Tier 1 — Urgent", "Tier 2 — Moderate", "Tier 3 — Self-sufficient"}
        self.assertTrue(set(df_out["priority_tier"]).issubset(valid))

    def test_rank_villages(self):
        from analysis.development import compute_development_scores, rank_villages
        df_out = compute_development_scores(self.df)
        ranked = rank_villages(df_out, by="revitalization_need", top_n=10)
        self.assertEqual(len(ranked), 10)


# =============================================================================
# K — Evaluation metrics
# =============================================================================
class TestEvaluationMetrics(unittest.TestCase):
    def test_internal_metrics_keys(self):
        from evaluation.metrics import internal_metrics
        X = np.random.randn(50, 8)
        labels = np.array([i % 5 for i in range(50)])
        m = internal_metrics(X, labels)
        self.assertIn("silhouette", m)
        self.assertIn("calinski_harabasz", m)
        self.assertIn("davies_bouldin", m)

    def test_perfect_ari(self):
        from evaluation.metrics import external_metrics
        y = np.array([0]*20 + [1]*20 + [2]*20)
        m = external_metrics(y, y)
        self.assertAlmostEqual(m["ari"], 1.0, places=4)

    def test_random_ari_near_zero(self):
        from evaluation.metrics import external_metrics
        np.random.seed(0)
        y_true = np.random.randint(0, 5, 200)
        y_pred = np.random.randint(0, 5, 200)
        m = external_metrics(y_pred, y_true)
        self.assertAlmostEqual(m["ari"], 0.0, delta=0.1)

    def test_silhouette_range(self):
        from evaluation.metrics import internal_metrics
        X = np.random.randn(30, 5)
        labels = np.array([i % 3 for i in range(30)])
        m = internal_metrics(X, labels)
        self.assertGreaterEqual(m["silhouette"], -1.0)
        self.assertLessEqual(m["silhouette"], 1.0)


# =============================================================================
# L — End-to-end smoke test
# =============================================================================
class TestEndToEnd(unittest.TestCase):
    def test_full_pipeline_runs(self):
        """Run the complete pipeline on 50 villages without error."""
        from config import Config
        from data.loader import load_data
        from data.preprocessing import preprocess
        from graph.builder import VillageGraphBuilder
        from models.spectral import SpectralClusterer
        from analysis.typology import label_clusters, typology_profile
        from analysis.development import compute_development_scores
        from evaluation.metrics import internal_metrics, external_metrics

        cfg = Config()
        cfg.data.n_villages = 50
        cfg.clustering.n_clusters = 5
        cfg.clustering.spectral_n_init = 5

        df, true_labels = load_data(n_villages=50, seed=99)
        df_proc, X, feat_names = preprocess(df, cfg.data.feature_groups)
        village_ids = list(df_proc.index)

        G = VillageGraphBuilder(cfg.graph).build(df_proc, X)
        self.assertGreater(G.number_of_edges(), 0)

        sc = SpectralClusterer(cfg.clustering)
        labels = sc.fit_predict(G, village_ids)
        self.assertEqual(len(labels), 50)

        mapping = label_clusters(df_proc, labels, village_ids, cfg.typology)
        self.assertEqual(len(mapping), 5)

        df_dev = compute_development_scores(df_proc)
        self.assertIn("revitalization_need", df_dev.columns)

        m_int = internal_metrics(X, labels)
        self.assertIn("silhouette", m_int)

        if true_labels is not None:
            m_ext = external_metrics(labels, true_labels)
            self.assertIn("ari", m_ext)
            self.assertGreater(m_ext["ari"], -0.1)

        print(f"\n  Smoke test passed — ARI={m_ext.get('ari', 'N/A'):.4f}  "
              f"Silhouette={m_int.get('silhouette', 'N/A'):.4f}")


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
