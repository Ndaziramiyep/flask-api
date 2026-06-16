from datetime import datetime
from flask import request, jsonify
from bson import ObjectId
from config.database import users_col, detections_col, notifications_col


def _safe_user(user):
    return {
        '_id': str(user['_id']),
        'name': user['name'],
        'email': user['email'],
        'role': user.get('role', 'user'),
        'avatar': user.get('avatar'),
        'notificationsEnabled': user.get('notificationsEnabled', True),
        'createdAt': user.get('createdAt', datetime.utcnow()).isoformat(),
    }


def get_all_users(current_user):
    page = max(1, int(request.args.get('page', 1)))
    limit = min(100, max(1, int(request.args.get('limit', 20))))
    skip = (page - 1) * limit
    search = request.args.get('search', '').strip()

    query = {}
    if search:
        query = {'$or': [
            {'name': {'$regex': search, '$options': 'i'}},
            {'email': {'$regex': search, '$options': 'i'}},
        ]}

    total = users_col.count_documents(query)
    users = list(users_col.find(query).sort('createdAt', -1).skip(skip).limit(limit))

    return jsonify({
        'success': True,
        'data': [_safe_user(u) for u in users],
        'pagination': {
            'page': page, 'limit': limit, 'total': total,
            'pages': max(1, (total + limit - 1) // limit),
        },
    }), 200


def get_user(current_user, user_id):
    try:
        user = users_col.find_one({'_id': ObjectId(user_id)})
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID'}), 400
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    scan_count = detections_col.count_documents({'userId': user_id})
    return jsonify({'success': True, 'data': {**_safe_user(user), 'scanCount': scan_count}}), 200


def update_user_role(current_user, user_id):
    data = request.get_json()
    role = data.get('role')
    if role not in ('user', 'admin'):
        return jsonify({'success': False, 'message': 'Role must be user or admin'}), 400
    try:
        users_col.update_one({'_id': ObjectId(user_id)}, {'$set': {'role': role, 'updatedAt': datetime.utcnow()}})
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID'}), 400
    return jsonify({'success': True, 'message': 'Role updated'}), 200


def delete_user(current_user, user_id):
    if user_id == str(current_user['_id']):
        return jsonify({'success': False, 'message': 'Cannot delete your own account'}), 400
    try:
        oid = ObjectId(user_id)
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID'}), 400
    detections_col.delete_many({'userId': user_id})
    notifications_col.delete_many({'userId': user_id})
    users_col.delete_one({'_id': oid})
    return jsonify({'success': True, 'message': 'User deleted'}), 200


def get_admin_stats(current_user):
    total_users = users_col.count_documents({})
    total_scans = detections_col.count_documents({})
    disease_pipeline = [
        {'$group': {'_id': '$disease', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}},
    ]
    disease_breakdown = list(detections_col.aggregate(disease_pipeline))
    recent_users = list(users_col.find({}).sort('createdAt', -1).limit(5))
    recent_scans = list(detections_col.find({}).sort('createdAt', -1).limit(5))

    def ser_scan(d):
        d = dict(d)
        d['_id'] = str(d['_id'])
        if 'createdAt' in d and hasattr(d['createdAt'], 'isoformat'):
            d['createdAt'] = d['createdAt'].isoformat()
        return d

    healthy = next((d['count'] for d in disease_breakdown if d['_id'] == 'Healthy'), 0)

    return jsonify({
        'success': True,
        'data': {
            'totalUsers': total_users,
            'totalScans': total_scans,
            'diseasesDetected': total_scans - healthy,
            'healthyScans': healthy,
            'diseaseBreakdown': [{'disease': d['_id'], 'count': d['count']} for d in disease_breakdown],
            'recentUsers': [_safe_user(u) for u in recent_users],
            'recentScans': [ser_scan(s) for s in recent_scans],
        },
    }), 200


def get_all_scans(current_user):
    page = max(1, int(request.args.get('page', 1)))
    limit = min(100, max(1, int(request.args.get('limit', 20))))
    skip = (page - 1) * limit
    search = request.args.get('disease', '').strip()

    query = {}
    if search:
        query['disease'] = {'$regex': search, '$options': 'i'}

    total = detections_col.count_documents(query)
    scans = list(detections_col.find(query).sort('createdAt', -1).skip(skip).limit(limit))

    def ser(d):
        d = dict(d)
        d['_id'] = str(d['_id'])
        if 'createdAt' in d and hasattr(d['createdAt'], 'isoformat'):
            d['createdAt'] = d['createdAt'].isoformat()
        return d

    return jsonify({
        'success': True,
        'data': [ser(s) for s in scans],
        'pagination': {
            'page': page, 'limit': limit, 'total': total,
            'pages': max(1, (total + limit - 1) // limit),
        },
    }), 200
