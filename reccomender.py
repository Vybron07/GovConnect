"""
GovConnect v2 — loads directly from pickle files built in Colab.
"""
import pickle, os, types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(__file__)

# ── Stubs so pickle can deserialise the saved classes ───────────────────
class EligibilityEngine:
    def check(self, scheme, profile):
        rules = scheme.get('eligibility', {})
        matched, missing = [], []
        total, passed = 0, 0

        if 'income_limit' in rules and profile.get('income'):
            total += 1
            if profile['income'] <= rules['income_limit']:
                passed += 1; matched.append(f"Income ≤ ₹{rules['income_limit']:,} ✓")
            else:
                missing.append(f"Income limit ₹{rules['income_limit']:,}/yr")

        if 'age_min' in rules and profile.get('age'):
            total += 1
            if profile['age'] >= rules['age_min']:
                passed += 1; matched.append(f"Age ≥ {rules['age_min']} ✓")
            else:
                missing.append(f"Min age {rules['age_min']} yrs")

        if 'age_max' in rules and profile.get('age'):
            total += 1
            if profile['age'] <= rules['age_max']:
                passed += 1; matched.append(f"Age ≤ {rules['age_max']} ✓")
            else:
                missing.append(f"Max age {rules['age_max']} yrs")

        if 'gender' in rules and profile.get('gender'):
            total += 1
            if profile['gender'].lower() == rules['gender'].lower():
                passed += 1; matched.append(f"Gender {rules['gender']} ✓")
            else:
                missing.append(f"For {rules['gender']} only")

        if 'categories' in rules and profile.get('category'):
            total += 1
            cats = [profile['category']] if isinstance(profile['category'], str) else profile['category']
            uc = [c.lower() for c in cats]
            sc = [c.lower() for c in rules['categories']]
            if any(u in s or s in u for u in uc for s in sc):
                passed += 1; matched.append('Category eligible ✓')
            else:
                missing.append(f"Need: {', '.join(rules['categories'][:3])}")

        if 'area' in rules and profile.get('area'):
            total += 1
            if profile['area'].lower() == rules['area'].lower():
                passed += 1; matched.append(f"Area {rules['area']} ✓")
            else:
                missing.append(f"{rules['area']} area only")

        if total == 0:
            return 0.5, ['Open to all eligible citizens'], []
        return round(passed / total, 2), matched, missing


class GovConnectPipeline:
    pass  # attributes filled by pickle.load


def _patch_module():
    """Inject stubs into __main__ so pickle.load can find the classes."""
    import sys
    main = sys.modules.get('__main__', types.ModuleType('__main__'))
    main.EligibilityEngine = EligibilityEngine
    main.GovConnectPipeline = GovConnectPipeline
    sys.modules['__main__'] = main


def detect_language(text):
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if deva == 0:
        return 'en'
    mr = sum(1 for m in ['ला','ची','चे','साठी','आहे'] if m in text)
    hi = sum(1 for m in ['है','हैं','का','के','की','में'] if m in text)
    return 'mr' if mr >= hi else 'hi'


def expand_query(query, glossary):
    expanded = query
    for term, english in glossary.items():
        if term in query:
            expanded += ' ' + english
    return expanded


# ── Main engine ─────────────────────────────────────────────────────────
class Engine:
    def __init__(self):
        _patch_module()

        # Load model artifact (embeddings + schemes + glossary + eval metrics)
        model_path = os.path.join(BASE, 'govconnect_model.pkl')
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)

        self.schemes    = artifact['schemes']
        self.embeddings = artifact['scheme_embeddings']   # (15, 384)
        self.glossary   = artifact['glossary']
        self.eval_metrics = {k: float(v) for k, v in artifact.get('eval_metrics', {}).items()}
        self.model_name = artifact.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.eligibility = EligibilityEngine()

        # Lazy-load ST model on first query
        self._st = None
        print(f"[GovConnect v2] Loaded {len(self.schemes)} schemes, embeddings {self.embeddings.shape}")

    def _get_st(self):
        if self._st is None:
            # Memory optimizations for Render Free Tier (512MB RAM)
            import os
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            import torch
            torch.set_num_threads(1)
            
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model_name)
            print("[GovConnect v2] ST model loaded.")
        return self._st

    def recommend(self, query, profile=None, top_k=5):
        profile  = profile or {}
        lang     = detect_language(query)
        expanded = expand_query(query, self.glossary)

        st   = self._get_st()
        qemb = st.encode([expanded], convert_to_numpy=True, show_progress_bar=False)
        sims = cosine_similarity(qemb, self.embeddings)[0]

        results = []
        for i, scheme in enumerate(self.schemes):
            es, matched, missing = self.eligibility.check(scheme, profile)
            final = sims[i] * (0.7 + 0.3 * es)
            results.append({
                'scheme':          scheme,
                'relevance_score': round(float(sims[i]), 4),
                'eligibility_score': es,
                'final_score':     round(float(final), 4),
                'matched_rules':   matched,
                'missing_rules':   missing,
                'detected_language': lang,
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)
        filtered = [r for r in results if r['relevance_score'] >= 0.01]
        return (filtered if len(filtered) >= 3 else results)[:top_k]


_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine
