# app.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from seed_data import initialize_data_files
from engine import SearchEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
initialize_data_files(BASE_DIR)

app = Flask(__name__, template_folder="templates")
CORS(app)

engine = SearchEngine(BASE_DIR)


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/api/documents", methods=["GET"])
def list_documents():
    role = request.args.get("role", "public")
    docs = []
    for did, doc in sorted(engine.documents.items()):
        docs.append({
            "id": did,
            "title": doc["title"],
            "author": doc["author"],
            "category": doc["category"],
            "year": doc["year"],
            "language": doc["language"],
            "acl": doc["acl"],
            "filename": doc["filename"],
            "is_deleted": did in engine.deleted_doc_ids,
            "is_auxiliary": did in engine.auxiliary_doc_ids,
            "accessible": not (doc["acl"] == "admin" and role != "admin")
        })
    return jsonify({"documents": docs})


@app.route("/api/documents/<int:doc_id>", methods=["GET"])
def get_document(doc_id: int):
    if doc_id not in engine.documents:
        return jsonify({"error": "Document not found"}), 404
    return jsonify(engine.documents[doc_id])


@app.route("/api/documents/add", methods=["POST"])
def add_document():
    data = request.get_json(force=True)
    doc = engine.add_document_dynamically(
        title=data.get("title", ""),
        author=data.get("author", ""),
        category=data.get("category", "General"),
        year=data.get("year", "2026"),
        acl=data.get("acl", "public"),
        body=data.get("body", "")
    )
    return jsonify({"status": "added", "document": doc})


@app.route("/api/documents/delete/<int:doc_id>", methods=["POST"])
def delete_document(doc_id: int):
    ok = engine.delete_document_dynamically(doc_id)
    if not ok:
        return jsonify({"error": "Document ID not found"}), 404
    return jsonify({"status": "deleted", "doc_id": doc_id})


@app.route("/api/documents/restore", methods=["POST"])
def restore_documents():
    engine.restore_deleted_documents()
    return jsonify({"status": "restored"})


@app.route("/api/boolean", methods=["POST"])
def boolean_search():
    data = request.get_json(force=True)
    query = data.get("query", "")
    options = data.get("options", {})
    try:
        res = engine.execute_boolean_query(query, options)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/advanced", methods=["POST"])
def advanced_search():
    data = request.get_json(force=True)
    query = data.get("query", "")
    options = data.get("options", {})
    try:
        res = engine.execute_advanced_query(query, options)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/click", methods=["POST"])
def record_click():
    data = request.get_json(force=True)
    engine.record_click(data.get("query", ""), int(data.get("doc_id", 0)))
    return jsonify({"status": "recorded"})


@app.route("/api/suggest", methods=["GET"])
def suggest_queries():
    q = request.args.get("q", "")
    return jsonify({"suggestions": engine.get_query_suggestions(q)})


@app.route("/api/stats", methods=["GET"])
def index_stats():
    return jsonify(engine.get_stats())


if __name__ == "__main__":
    print("Starting WSM Search Engine on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)