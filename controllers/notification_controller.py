from datetime import datetime
from flask import request, jsonify
from bson import ObjectId
from config.database import notifications_col


def _serialize(doc: dict) -> dict:
    doc = dict(doc)
    doc['_id'] = str(doc['_id'])
    if 'createdAt' in doc:
        doc['createdAt'] = doc['createdAt'].isoformat()
    return doc


def get_notifications(current_user):
    page = max(1, int(request.args.get('page', 1)))
    limit = min(50, int(request.args.get('limit', 20)))
    skip = (page - 1) * limit

    query = {'userId': str(current_user['_id'])}
    total = notifications_col.count_documents(query)
    records = list(notifications_col.find(query).sort('createdAt', -1).skip(skip).limit(limit))

    return jsonify({
        'success': True,
        'data': [_serialize(r) for r in records],
        'pagination': {
            'page': page, 'limit': limit, 'total': total,
            'pages': (total + limit - 1) // limit,
        },
    }), 200


def get_unread_count(current_user):
    count = notifications_col.count_documents({'userId': str(current_user['_id']), 'isRead': False})
    return jsonify({'success': True, 'data': {'count': count}}), 200


def mark_as_read(current_user, notification_id):
    try:
        result = notifications_col.find_one_and_update(
            {'_id': ObjectId(notification_id), 'userId': str(current_user['_id'])},
            {'$set': {'isRead': True}},
            return_document=True,
        )
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID'}), 400

    if not result:
        return jsonify({'success': False, 'message': 'Notification not found'}), 404

    return jsonify({'success': True, 'data': _serialize(result)}), 200


def mark_all_as_read(current_user):
    notifications_col.update_many(
        {'userId': str(current_user['_id']), 'isRead': False},
        {'$set': {'isRead': True}},
    )
    return jsonify({'success': True, 'message': 'All notifications marked as read'}), 200


def delete_notification(current_user, notification_id):
    try:
        result = notifications_col.find_one_and_delete(
            {'_id': ObjectId(notification_id), 'userId': str(current_user['_id'])},
        )
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid ID'}), 400

    if not result:
        return jsonify({'success': False, 'message': 'Notification not found'}), 404

    return jsonify({'success': True, 'message': 'Notification deleted'}), 200
