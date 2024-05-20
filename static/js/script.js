document.addEventListener("DOMContentLoaded", function () {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("user_input");
    const sendButton = document.getElementById("send_button");

    sendButton.addEventListener("click", function () {
        const userMessage = userInput.value;
        if (userMessage.trim() !== "") {
            appendMessage("user", userMessage);
            fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: userMessage }),
            })
                .then(response => response.json())
                .then(data => {
                    appendMessage("bot", data.reply);
                });
        }
        userInput.value = "";
    });

    function appendMessage(role, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        messageElement.classList.add(role);
        messageElement.textContent = message;
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});
