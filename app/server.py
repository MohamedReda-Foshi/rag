from flask import Flask, request, jsonify, render_template_string
from core_rag import query
import re

def sanitize_input(text: str) -> str:
    # Remove max length
    text = text[:500]
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove prompt injection attempts
    blacklist = [
        "ignore previous", "ignore above", "disregard",
        "you are now", "new instructions", "system prompt",
        "forget everything", "act as", "jailbreak"
    ]
    lower = text.lower()
    for phrase in blacklist:
        if phrase in lower:
            return None
    return text.strip()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024  # max 1KB request
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>JS RAG Assistant</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 40px auto; padding: 20px; }
        input { width: 80%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }
        #answer { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 8px; white-space: pre-wrap; }
        #sources { margin-top: 10px; font-size: 13px; color: #666; }
    </style>
</head>
<body>
    <h1>📖 JavaScript RAG Assistant</h1>
    <input type="text" id="question" placeholder="Ask about JavaScript...">
    <button onclick="ask()">Ask</button>
    <div id="answer"></div>
    <div id="sources"></div>

    <script>
        async function ask() {
            const q = document.getElementById("question").value;
            document.getElementById("answer").innerText = "⏳ Thinking...";
            const res = await fetch("/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question: q})
            });
            const data = await res.json();
            document.getElementById("answer").innerText = data.answer;
            document.getElementById("sources").innerText = 
                "Sources: " + data.sources.map(s => `${s.source} (page ${s.page})`).join(", ");
        }

        document.getElementById("question").addEventListener("keypress", e => {
            if (e.key === "Enter") ask();
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400
        
        # Sanitize
    question = sanitize_input(data["question"])
    if not question:
            return jsonify({"error": "Invalid or suspicious input"}), 400
        
    result = query(question)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)