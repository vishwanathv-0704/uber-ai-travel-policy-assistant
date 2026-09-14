import json
import requests

from src.tools import (
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement
)

from src.rag import rag_answer


# ============================================================
# CONFIGURATION
# ============================================================

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
# AGENT DECISION
# ============================================================

def select_action(user_query, employee_id=None, memory_context=""):

    prompt = f"""
You are an enterprise travel policy assistant.

Your job is to decide what action is required to answer
the user's question.

AVAILABLE ACTIONS:

1. rag

Use RAG for general company policy questions.

Examples:
- What is the standard travel limit in India?
- What are the airport travel rules?
- What is the cancellation policy?
- What happens if I exceed the travel limit?

2. check_employee_eligibility

Use this for employee eligibility questions.

Example:
- Is EMP001 eligible for business travel?

3. validate_trip

Use this when the user asks whether a specific trip is
allowed or requires approval.

4. calculate_reimbursement

Use this when the user asks how much an expense can
be reimbursed.

5. complex_trip

Use this when answering the question requires multiple
steps, especially:

Employee Eligibility
+
RAG Policy Retrieval
+
Trip Validation

Example:
"Can EMP001 take a 2500 airport trip?"

6. none

Use when none of the above applies.

IMPORTANT:

A complex trip question MUST use:

check_employee_eligibility
+
rag
+
validate_trip

Do NOT answer a complex trip question using only one tool.

CONVERSATION CONTEXT:

{memory_context}

EMPLOYEE ID FROM FORM:

{employee_id}

USER QUESTION:

{user_query}

Return ONLY valid JSON.

For RAG:

{{
    "action": "rag",
    "arguments": {{}}
}}

For eligibility:

{{
    "action": "check_employee_eligibility",
    "arguments": {{
        "employee_id": "EMP001"
    }}
}}

For trip validation:

{{
    "action": "validate_trip",
    "arguments": {{
        "employee_id": "EMP001",
        "trip_type": "Airport",
        "amount": 2500,
        "time": "22:30"
    }}
}}

For reimbursement:

{{
    "action": "calculate_reimbursement",
    "arguments": {{
        "employee_id": "EMP001",
        "amount": 2500,
        "expense_type": "Airport Travel"
    }}
}}

For complex trip:

{{
    "action": "complex_trip",
    "arguments": {{
        "employee_id": "EMP001",
        "trip_type": "Airport",
        "amount": 2500,
        "time": "22:30"
    }}
}}

For none:

{{
    "action": "none",
    "arguments": {{}}
}}

Only include information actually present in the question
or employee ID field.

Do NOT invent an employee ID.

Do NOT invent an amount.

Do NOT invent a time.
"""

    result = ask_ollama(prompt)

    try:

        # Remove accidental markdown fences
        result = result.strip()

        if result.startswith("```"):
            result = result.replace("```json", "")
            result = result.replace("```", "")
            result = result.strip()

        return json.loads(result)

    except json.JSONDecodeError:

        return {
            "action": "none",
            "arguments": {}
        }


# ============================================================
# SINGLE TOOL EXECUTION
# ============================================================

def execute_tool(action, arguments):

    if action == "check_employee_eligibility":

        return check_employee_eligibility(
            arguments.get("employee_id")
        )

    elif action == "validate_trip":

        try:

            return validate_trip(
                arguments.get("employee_id"),
                arguments.get("trip_type"),
                arguments.get("amount"),
                arguments.get("time")
            )

        except Exception as e:

            return {
                "status": "Validation Error",
                "message": str(e)
            }

    elif action == "calculate_reimbursement":

        return calculate_reimbursement(
            arguments.get("employee_id"),
            arguments.get("amount"),
            arguments.get("expense_type")
        )

    return {
        "status": "No Tool",
        "message": "No suitable tool was found."
    }


# ============================================================
# FINAL ANSWER FOR NORMAL TOOL
# ============================================================

def generate_final_answer(
    user_query,
    action,
    tool_result,
    memory_context=""
):

    prompt = f"""
You are an enterprise travel policy assistant.

Answer the user's question using ONLY the information
provided in the tool result.

IMPORTANT RULES:

- Do not invent information.
- Do not change numbers.
- Do not invent policy limits.
- Clearly explain the result.
- Keep the answer concise.
- Mention the employee ID when available.
- If approval is required, clearly say so.
- If the employee is Not Found or Not Eligible,
  clearly explain that.
- Do not claim a trip is allowed if approval is required.

Previous conversation context:

{memory_context}

User question:

{user_query}

Action used:

{action}

Tool result:

{json.dumps(tool_result, indent=2)}

Now provide a concise final answer.
"""

    return ask_ollama(prompt)


# ============================================================
# FINAL ANSWER FOR COMPLEX TRIP
# ============================================================

def generate_complex_answer(
    user_query,
    employee_result,
    rag_result,
    trip_result,
    memory_context=""
):

    prompt = f"""
You are an enterprise travel policy assistant.

Answer the user's question using ONLY the information
provided below.

The user asked:

{user_query}

Previous conversation:

{memory_context}

EMPLOYEE ELIGIBILITY RESULT:

{json.dumps(employee_result, indent=2)}

POLICY / RAG RESULT:

{json.dumps(rag_result, indent=2)}

TRIP VALIDATION RESULT:

{json.dumps(trip_result, indent=2)}

IMPORTANT RULES:

1. Do not invent information.
2. Do not change any numbers.
3. Clearly state whether the employee is eligible.
4. Clearly state the relevant policy limit.
5. Clearly state whether the trip is allowed,
   requires approval, or is not eligible.
6. If approval is required, explicitly say:
   "Approval is required."
7. If the employee is not eligible, clearly state that.
8. Use the policy source provided by RAG.
9. Do not claim reimbursement unless the data supports it.
10. Keep the answer concise and professional.

Return only the final answer.
"""

    return ask_ollama(prompt)


# ============================================================
# MAIN AGENT
# ============================================================

def run_agent(
    user_query,
    employee_id=None,
    memory=None
):

    print("\n" + "=" * 70)
    print("USER REQUEST")
    print("=" * 70)

    print(user_query)

    # --------------------------------------------------------
    # MEMORY CONTEXT
    # --------------------------------------------------------

    memory_context = ""

    if memory is not None:

        try:
            memory_context = memory.get_context()
        except Exception:
            memory_context = ""

    # --------------------------------------------------------
    # AGENT DECISION
    # --------------------------------------------------------

    decision = select_action(
        user_query,
        employee_id,
        memory_context
    )

    print("\n" + "=" * 70)
    print("AGENT DECISION")
    print("=" * 70)

    print(
        json.dumps(
            decision,
            indent=2
        )
    )

    action = decision.get(
        "action",
        "none"
    )

    arguments = decision.get(
        "arguments",
        {}
    )

    # If employee ID was entered in Flask but not extracted
    # by the LLM, use the form value.

    if employee_id and not arguments.get("employee_id"):

        arguments["employee_id"] = employee_id

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if action == "rag":

        result = rag_answer(
            user_query
        )

        print("\n" + "=" * 70)
        print("RAG RESULT")
        print("=" * 70)

        print(result["answer"])

        print("\nSources:")

        for source in result["sources"]:
            print("-", source)

        final_answer = (
            result["answer"]
            + "\n\nPolicy Sources: "
            + ", ".join(result["sources"])
        )

    # --------------------------------------------------------
    # COMPLEX TRIP
    # --------------------------------------------------------

    elif action == "complex_trip":

        # -----------------------------------------------
        # STEP 1: Employee Eligibility
        # -----------------------------------------------

        print("\n" + "=" * 70)
        print("STEP 1 - EMPLOYEE ELIGIBILITY")
        print("=" * 70)

        employee_result = check_employee_eligibility(
            arguments.get("employee_id")
        )

        print(
            json.dumps(
                employee_result,
                indent=2
            )
        )

        # -----------------------------------------------
        # STEP 2: RAG POLICY RETRIEVAL
        # -----------------------------------------------

        print("\n" + "=" * 70)
        print("STEP 2 - RAG POLICY RETRIEVAL")
        print("=" * 70)

        rag_query = (
            f"What is the policy limit and approval requirement "
            f"for a {arguments.get('trip_type')} trip in "
            f"{employee_result.get('country', '')}?"
        )

        rag_result = rag_answer(
            rag_query
        )

        print(rag_result["answer"])

        print("\nSources:")

        for source in rag_result["sources"]:
            print("-", source)

        # -----------------------------------------------
        # STEP 3: TRIP VALIDATION
        # -----------------------------------------------

        print("\n" + "=" * 70)
        print("STEP 3 - TRIP VALIDATION")
        print("=" * 70)

        trip_result = execute_tool(
            "validate_trip",
            arguments
        )

        print(
            json.dumps(
                trip_result,
                indent=2
            )
        )

        # -----------------------------------------------
        # STEP 4: COMBINE EVERYTHING
        # -----------------------------------------------

        final_answer = generate_complex_answer(
            user_query,
            employee_result,
            rag_result,
            trip_result,
            memory_context
        )

        # Add sources
        final_answer += (
            "\n\nPolicy Sources: "
            + ", ".join(rag_result["sources"])
        )

    # --------------------------------------------------------
    # NONE
    # --------------------------------------------------------

    elif action == "none":

        final_answer = (
            "I could not determine a suitable "
            "policy or tool for this request."
        )

    # --------------------------------------------------------
    # NORMAL TOOL
    # --------------------------------------------------------

    else:

        tool_result = execute_tool(
            action,
            arguments
        )

        print("\n" + "=" * 70)
        print("TOOL RESULT")
        print("=" * 70)

        print(
            json.dumps(
                tool_result,
                indent=2
            )
        )

        final_answer = generate_final_answer(
            user_query,
            action,
            tool_result,
            memory_context
        )

    # --------------------------------------------------------
    # UPDATE MEMORY
    # --------------------------------------------------------

    if memory is not None:

        try:

            memory.add_user_message(
                user_query
            )

            memory.add_assistant_message(
                final_answer
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(final_answer)

    return final_answer


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("AGENT TEST")
    print("=" * 70)

    # Test 1: RAG
    print("\n\nTEST 1")
    run_agent(
        "What is the standard travel limit in India?"
    )

    # Test 2: Employee tool
    print("\n\nTEST 2")
    run_agent(
        "Is EMP001 eligible for business travel?"
    )

    # Test 3: Complex agent workflow
    print("\n\nTEST 3")
    run_agent(
        "Can EMP001 take an airport trip costing 2500 at 22:30?"
    )
