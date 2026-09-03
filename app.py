import os, traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

_engine = None
def engine():
    global _engine
    if _engine is None:
        from recommender import get_engine
        _engine = get_engine()
    return _engine

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data    = request.get_json(force=True)
        query   = data.get('query', '').strip()
        profile = data.get('profile', {})
        top_k   = int(data.get('top_k', 5))
        if not query:
            return jsonify({'error': 'query is required'}), 400

        results = engine().recommend(query, profile=profile, top_k=top_k)
        out = []
        for r in results:
            s = r['scheme']
            out.append({
                'id':                s['id'],
                'name':              s['name'],
                'name_mr':           s.get('name_mr', ''),
                'name_hi':           s.get('name_hi', ''),
                'department':        s['department'],
                'category':          s['category'],
                'description':       s['description'],
                'description_mr':    s.get('description_mr', ''),
                'description_hi':    s.get('description_hi', ''),
                'benefits':          s.get('benefits', ''),
                'documents':         s.get('documents', []),
                'portal':            s.get('portal', '#'),
                'relevance_score':   r['relevance_score'],
                'eligibility_score': r['eligibility_score'],
                'final_score':       r['final_score'],
                'matched_rules':     r['matched_rules'],
                'missing_rules':     r['missing_rules'],
                'detected_language': r['detected_language'],
            })
        return jsonify({'results': out, 'count': len(out)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/schemes')
def list_schemes():
    try:
        eng = engine()
        lite = [{'id': s['id'], 'name': s['name'], 'department': s['department'],
                 'category': s['category'], 'benefits': s.get('benefits', ''),
                 'portal': s.get('portal', '#')} for s in eng.schemes]
        return jsonify({'schemes': lite, 'total': len(lite)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scheme/<sid>')
def get_scheme(sid):
    try:
        for s in engine().schemes:
            if s['id'] == sid:
                return jsonify(s)
        return jsonify({'error': 'Not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics')
def metrics():
    try:
        return jsonify(engine().eval_metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'project': 'SIH26129 GovConnect v2'})

if __name__ == '__main__':
    print('[GovConnect v2] Pre-loading engine...')
    engine()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
