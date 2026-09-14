import json
import requests

from src.tools import (
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement
)


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


# ============================================================
# OLLAMA HELPER
# ============================================================

def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# ============================================================
# TOOL SELECTION
# ============================================================

def select_tool(user_query):
    """
    Ask the LLM which tool should be used and extract
    the required parameters.
    """

    prompt = f"""
You are an enterprise travel policy assistant.

You have access to exactly these tools:

1. check_employee_eligibility
   - Use when the user asks whether an employee is eligible.

2. validate_trip
   - Use when the user asks whether a business trip is allowed,
     needs approval, or is within the travel spending limit.

3. calculate_reimbursement
   - Use when the user asks how much of an expense can be reimbursed.

Read the user's request and return ONLY valid JSON.

JSON format:

{{
    "tool": "tool_name",
    "arguments": {{
        "employee_id": "...",
        "trip_type": "...",
        "amount": 0,
        "time": "...",
        "expense_type": "..."
    }}
}}

Only include arguments that are relevant to the selected tool.

If the request does not match any tool, return:

{{
    "tool": "none",
    "arguments": {{}}
}}

User request:
{user_query}
"""

    result = ask_ollama(prompt)

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {
            "tool": "none",
            "arguments": {}
        }


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(tool_name, arguments):

    if tool_name == "check_employee_eligibility":

        return check_employee_eligibility(
            arguments.get("employee_id")
        )

    elif tool_name == "validate_trip":

        return validate_trip(
            arguments.get("employee_id"),
            arguments.get("trip_type"),
            arguments.get("amount"),
            arguments.get("time")
        )

    elif tool_name == "calculate_reimbursement":

        return calculate_reimbursement(
            arguments.get("employee_id"),
            arguments.get("amount"),
            arguments.get("expense_type")
        )

    else:
        return {
            "status": "No Tool",
            "message": "No suitable tool was found for this request."
        }


# ============================================================
# FINAL RESPONSE GENERATION
# ============================================================

def generate_final_answer(user_query, tool_name, tool_result):

    prompt = f"""
You are an enterprise travel policy assistant.

Answer the user's question using the tool result below.

IMPORTANT RULES:
- Do not invent information.
- Do not change numbers returned by the tool.
- Clearly explain the result.
- Keep the answer concise.
- Mention the relevant employee ID when available.
- If approval is required, clearly say so.
- If the tool says Not Found or Not Eligible, clearly explain that.

User question:
{user_query}

Tool used:
{tool_name}

Tool result:
{json.dumps(tool_result, indent=2)}

Now provide the final answer to the user.
"""

    return ask_ollama(prompt)


# ============================================================
# MAIN AGENT
# ============================================================

def run_agent(user_query):

    print("\n" + "=" * 70)
    print("USER REQUEST")
    print("=" * 70)
    print(user_query)

    # Step 1: LLM decides which tool to use
    decision = select_tool(user_query)

    print("\n" + "=" * 70)
    print("LLM TOOL DECISION")
    print("=" * 70)
    print(json.dumps(decision, indent=2))

    tool_name = decision.get("tool")
    arguments = decision.get("arguments", {})

    # Step 2: Execute selected tool
    tool_result = execute_tool(
        tool_name,
        arguments
    )

    print("\n" + "=" * 70)
    print("TOOL RESULT")
    print("=" * 70)
    print(json.dumps(tool_result, indent=2))

    # Step 3: LLM converts tool result into final answer
    final_answer = generate_final_answer(
        user_query,
        tool_name,
        tool_result
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(final_answer)

    return final_answer
