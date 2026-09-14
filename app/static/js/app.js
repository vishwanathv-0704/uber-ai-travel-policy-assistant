async function askAssistant() {

    const employeeId =
        document.getElementById("employeeId").value.trim();

    const question =
        document.getElementById("question").value.trim();

    const responseBox =
        document.getElementById("response");

    const historyBox =
        document.getElementById("history");


    if (!employeeId) {
        responseBox.innerText = "Please enter an employee ID.";
        return;
    }

    if (!question) {
        responseBox.innerText = "Please enter a question.";
        return;
    }


    responseBox.innerText = "Thinking...";


    try {

        const response = await fetch("/ask", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                employee_id: employeeId,
                question: question
            })

        });


        const data = await response.json();


        if (data.success) {

            responseBox.innerText = data.answer + "\n\nPolicy Source:" + data.source;

            historyBox.innerText =
                data.history;

            document.getElementById("question").value = "";

        } else {

            responseBox.innerText =
                "Error: " + data.error;

        }

    } catch (error) {

        responseBox.innerText =
            "Unable to connect to the AI assistant.";

    }
}


async function clearConversation() {

    try {

        const response = await fetch("/clear", {
            method: "POST"
        });

        const data = await response.json();

        if (data.success) {

            document.getElementById("response").innerText =
                "Conversation cleared.";

            document.getElementById("history").innerText =
                "No conversation yet.";

        }

    } catch (error) {

        document.getElementById("response").innerText =
            "Unable to clear conversation.";

    }
}


async function checkHealth() {

    try {

        const response = await fetch("/health");

        const data = await response.json();

        document.getElementById("status").innerText =
            "🟢 " + data.status;

    } catch (error) {

        document.getElementById("status").innerText =
            "🔴 Service unavailable";

    }
}


checkHealth();
