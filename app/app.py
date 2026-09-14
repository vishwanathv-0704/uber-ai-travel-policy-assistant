from flask import Flask, render_template, request, jsonify
import sys
import os

# Allow imports from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.agent import run_agent
from src.memory import ConversationMemory


app = Flask(__name__)

# Conversation memory
memory = ConversationMemory()


@app.route("/", methods=["GET"])
def home():
    """Render the main dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Uber AI Travel Policy Assistant"
    })


@app.route("/ask", methods=["POST"])
def ask():
    """Process a user question through the AI agent."""

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No request data provided."
            }), 400

        employee_id = data.get("employee_id", "").strip()
        question = data.get("question", "").strip()

        if not employee_id:
            return jsonify({
                "success": False,
                "error": "Please provide an employee ID."
            }), 400

        if not question:
            return jsonify({
                "success": False,
                "error": "Please enter a question."
            }), 400

        # Add employee context to the question
        conversation_context = memory.get_context()

        user_query = (
        	f"Employee ID: {employee_id}\n\n"
                f"Previous conversation:\n"
                f"{conversation_context}\n\n"
                f"Current user question:\n"
                f"{question}"
        )

        # Run the existing AI agent
        answer = run_agent(user_query)

        # Store conversation
        memory.add_user_message(question)
        memory.add_assistant_message(str(answer))

        # return jsonify({
        #     "success": True,
        #     "employee_id": employee_id,
        #     "question": question,
        #     "answer": str(answer),
        #     "history": memory.get_context()
        # })
        # Determine the relevant source for the response
        question_lower = question.lower()

        if "cost" in question_lower or "amount" in question_lower or "limit" in question_lower:
            source = "travel_policy_india.txt + validate_trip tool"
        elif "airport" in question_lower:
            source = "airport_policy.txt + employee eligibility tool"
        elif "eligible" in question_lower or "eligibility" in question_lower:
            source = "employees.csv + employee eligibility tool"
        else:
            source = "AI Travel Policy Assistant knowledge base"

        return jsonify({
            "success": True,
            "employee_id": employee_id,
            "question": question,
            "answer": str(answer),
            "source": source,
            "history": memory.get_context()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": "Something went wrong while processing your request."
        }), 500


@app.route("/clear", methods=["POST"])
def clear():
    """Clear conversation memory."""

    try:
        # Re-create memory object to start a fresh conversation
        global memory
        memory = ConversationMemory()

        return jsonify({
            "success": True,
            "message": "Conversation cleared."
        })

    except Exception:
        return jsonify({
            "success": False,
            "error": "Unable to clear conversation."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
