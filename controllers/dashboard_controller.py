from datetime import datetime, timedelta
from flask import jsonify
from config.database import detections_col


def get_stats(current_user):
    user_id = str(current_user['_id'])
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    pipeline_breakdown = [
        {'$match': {'userId': user_id}},
        {'$group': {'_id': '$disease', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}},
    ]

    pipeline_by_day = [
        {'$match': {'userId': user_id, 'createdAt': {'$gte': thirty_days_ago}}},
        {'$group': {
            '_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$createdAt'}},
            'count': {'$sum': 1},
        }},
        {'$sort': {'_id': 1}},
    ]

    total_scans = detections_col.count_documents({'userId': user_id})
    disease_breakdown_raw = list(detections_col.aggregate(pipeline_breakdown))
    recent_scans_raw = list(detections_col.find({'userId': user_id}).sort('createdAt', -1).limit(7))
    scans_by_day_raw = list(detections_col.aggregate(pipeline_by_day))

    healthy_count = next((d['count'] for d in disease_breakdown_raw if d['_id'] == 'Healthy'), 0)
    diseases_detected = total_scans - healthy_count

    breakdown = [
        {
            'disease': item['_id'],
            'count': item['count'],
            'percentage': round(item['count'] / total_scans * 100) if total_scans > 0 else 0,
        }
        for item in disease_breakdown_raw
    ]

    most_common = next((d['disease'] for d in breakdown if d['disease'] != 'Healthy'), None)

    def serialize(doc):
        doc = dict(doc)
        doc['_id'] = str(doc['_id'])
        if 'createdAt' in doc:
            doc['createdAt'] = doc['createdAt'].isoformat()
        return doc

    return jsonify({
        'success': True,
        'data': {
            'totalScans': total_scans,
            'diseasesDetected': diseases_detected,
            'healthyCount': healthy_count,
            'recentScans': [serialize(r) for r in recent_scans_raw],
            'diseaseBreakdown': breakdown,
            'scansByDay': [{'date': d['_id'], 'count': d['count']} for d in scans_by_day_raw],
            'mostCommonDisease': most_common,
        },
    }), 200
