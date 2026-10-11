import unittest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

class TestCivicFindSystem(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_homepage(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CIVIC', response.data)
        self.assertIn(b'FIND', response.data)
        self.assertIn(b'Okapi BM25', response.data)

    def test_search_bm25(self):
        payload = {
            "query": "scholarships for engineering college students in Maharashtra",
            "model": "bm25",
            "use_expansion": True,
            "top_k": 5
        }
        response = self.client.post('/api/search', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        first_doc = data["results"][0]
        self.assertIn("SCH-EDU", first_doc["doc_id"])
        print("\n[TEST] BM25 Top 1 Result:", first_doc["scheme"]["name"], f"({first_doc['relevance_pct']}%)")

    def test_search_vsm(self):
        payload = {
            "query": "farming subsidies and crop insurance",
            "model": "vsm",
            "use_expansion": False,
            "top_k": 5
        }
        response = self.client.post('/api/search', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        print("[TEST] VSM Top 1 Result:", data["results"][0]["scheme"]["name"])

    def test_search_boolean(self):
        payload = {
            "query": "farmer AND insurance",
            "model": "boolean",
            "top_k": 5
        }
        response = self.client.post('/api/search', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("results", data)
        print("[TEST] Boolean matches count:", len(data["results"]))

    def test_search_semantic(self):
        payload = {
            "query": "monetary assistance for post matriculation studies",
            "model": "semantic",
            "top_k": 5
        }
        response = self.client.post('/api/search', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("results", data)
        print("[TEST] Semantic Top 1 Result:", data["results"][0]["scheme"]["name"])

    def test_search_hybrid(self):
        payload = {
            "query": "solar power subsidies for households",
            "model": "hybrid",
            "top_k": 5
        }
        response = self.client.post('/api/search', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("results", data)
        print("[TEST] Hybrid Top 1 Result:", data["results"][0]["scheme"]["name"])

    def test_rocchio_relevance_feedback(self):
        payload = {
            "original_query": "farmer crop",
            "relevant_doc_ids": ["SCH-AGR-001", "SCH-AGR-003"],
            "non_relevant_doc_ids": ["SCH-EDU-001"],
            "top_k": 5
        }
        response = self.client.post('/api/relevance-feedback', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("results", data)
        self.assertIn("feedback_info", data)
        print("[TEST] Rocchio added terms:", data["feedback_info"]["added_terms"])

    def test_index_stats_and_postings(self):
        res1 = self.client.get('/api/index/stats')
        self.assertEqual(res1.status_code, 200)
        stats = json.loads(res1.data)
        self.assertGreater(stats["num_documents"], 100)

        res2 = self.client.get('/api/index/term/scholarship')
        self.assertEqual(res2.status_code, 200)
        term_data = json.loads(res2.data)
        self.assertEqual(term_data["term"], "scholarship")
        self.assertGreater(term_data["document_frequency"], 0)
        print(f"[TEST] 'scholarship' df={term_data['document_frequency']}, idf={term_data['idf']}")

    def test_evaluation_benchmark(self):
        res = self.client.get('/api/evaluate')
        self.assertEqual(res.status_code, 200)
        eval_data = json.loads(res.data)
        self.assertIn("models_summary", eval_data)
        for model in eval_data["models_summary"]:
            print(f"[TEST EVAL] {model['model_name']}: MAP={model['map']}, NDCG@10={model['mean_ndcg@10']}, P@5={model['mean_p@5']}")

if __name__ == '__main__':
    unittest.main()
