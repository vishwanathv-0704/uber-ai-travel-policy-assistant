# Uber for Business – AI Travel & Policy Assistant

An end-to-end AI-powered Travel & Policy Assistant built using RAG,
LLM-based agentic decision making, tool calling, conversational memory,
MCP, and Flask.

> All employee and policy data used in this project is fictional training
> data and does not contain Uber internal or private information.

---

## 1. Project Overview

The AI Travel & Policy Assistant helps employees understand company
travel policies and validate travel-related requests.

The system can:

- Answer questions from company travel policies
- Retrieve relevant policy information using RAG
- Provide policy source attribution
- Check employee business-travel eligibility
- Validate travel requests against policy limits
- Calculate reimbursable expenses
- Decide whether to use RAG or tools
- Combine multiple tools and RAG for complex requests
- Maintain conversational context
- Expose tools through an MCP server
- Provide a browser-based Flask dashboard
- Handle unsupported questions without inventing policy information

---

## 2. System Architecture

```text
                         USER
                           |
                           v
                    Flask Dashboard
                           |
                           v
                       AI Agent
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
            RAG          Tools         Memory
             |             |
             |       +-----+------+----------------+
             |       |            |                |
             v       v            v                v
        BGE-large  Employee    Trip          Reimbursement
             |     Eligibility Validation      Calculator
             |
             v
           FAISS
             |
             v
       Policy Documents
             |
             v
           Ollama
             |
             v
        Final Response
             |
             v
       Source Attribution


# Uber for Business – AI Travel & Policy Assistant

An AI-powered travel and policy assistant built using RAG, LLMs, agentic
tool calling, conversational memory, MCP, and Flask.

## Features

- Policy question answering using RAG
- FAISS vector search
- BGE-large embeddings
- Local LLM using Ollama
- Employee eligibility checking
- Trip validation against policy limits
- Expense reimbursement calculation
- Agent-based tool selection
- Multi-step RAG + tool workflows
- Conversational memory
- MCP server for exposing tools
- Flask web dashboard
- Policy source attribution
- Hallucination prevention

## Tech Stack

- Python
- Ollama (`llama3.2:3b`)
- FAISS
- `BAAI/bge-large-en-v1.5`
- LangChain Text Splitter
- Flask
- MCP
- HTML / CSS / JavaScript
- Pandas

## Architecture

```text
User
  |
  v
Flask Dashboard
  |
  v
AI Agent
  |
  +------ RAG ------> FAISS ---> Policy Documents
  |
  +------ Tools ----> Eligibility
  |                  Trip Validation
  |                  Reimbursement
  |
  +------ Memory ---> Conversation History
  |
  v
Final Response



RAG Pipeline
Policy Documents
      |
      v
Cleaning & Chunking
      |
      v
BGE-large Embeddings
      |
      v
FAISS Vector Store
      |
      v
Relevant Policy Chunks
      |
      v
Ollama LLM
      |
      v
Grounded Answer + Source
The policy documents are chunked using a chunk size of 500 and overlap of 100.
FAISS uses normalized embeddings with IndexFlatIP for similarity search.
Agent Tools
The agent can select between RAG and the following tools:
check_employee_eligibility(employee_id)

validate_trip(employee_id, trip_type, amount, time)

calculate_reimbursement(employee_id, amount, expense_type)
For complex requests, the agent can combine employee eligibility, policy retrieval, and trip validation before generating the final response.
Conversational Memory
The assistant maintains short-term conversation history so that follow-up questions can use previous context.
Example:
User: Can I take an airport trip?

Assistant: Airport trips are allowed according to policy.

User: What if it costs 2500?

Assistant: The trip exceeds the applicable policy limit and requires
approval.
MCP
The project includes an MCP server exposing the three business tools:
Employee Eligibility
Trip Validation
Reimbursement Calculation
Flask Application
Available routes:
GET  /
POST /ask
POST /clear
GET  /health
The Flask application acts as the UI/API layer while the AI logic remains inside the backend modules.
Project Structure
uber-ai-policy-assistant/
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── data/
│   ├── company_policy/
│   └── employees.csv
│
├── mcp/
│   └── server.py
│
├── notebooks/
│
├── src/
│   ├── agent.py
│   ├── ingestion.py
│   ├── rag.py
│   ├── tools.py
│   └── memory.py
│
├── tests/
│   └── test_cases.md
│
├── README.md
└── requirements.txt
Running the Project
Activate environment
source venv/bin/activate
Start MCP Server
PYTHONPATH=. python mcp/server.py
Start Flask
PYTHONPATH=. python app/app.py
Open:
http://127.0.0.1:5000
Make sure Ollama is running with:
llama3.2:3b
Testing
The system is tested across:
RAG policy questions
Employee eligibility
Trip validation
Reimbursement
Conversational memory
Complex multi-tool workflows
Hallucination prevention
Error handling
Unsupported questions should not result in fabricated policy information. The assistant should state when the required information cannot be found in the available policy documents.
Limitations
Uses fictional employee and policy data
Memory is currently in-memory
Authentication is not implemented
Designed as a training/demo application
Local LLM performance depends on available hardware
Future Improvements
Persistent memory
Authentication and authorization
More policy documents and countries
Better retrieval evaluation
Policy document upload
Docker deployment
Improved agent guardrails
Production database integration
Conclusion
This project demonstrates an end-to-end enterprise AI assistant combining RAG, LLMs, agentic tool calling, memory, MCP, and Flask into a single application.