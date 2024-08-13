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
                    const formattedResponse = data.reply.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.+?)\*/g, '<em>$1</em>')
                        .replace(/(\d+\.\s+)/g, '<br>$1')
                        .replace(/\n/g, '<br>');
                    appendMessage(formattedResponse);
                })
                .catch(error => {
                    console.error('Error fetching response:', error);
                    appendMessage('Error generating response. Please try again later.');
                });
        }
        userInput.value = "";
    });

    function appendMessage(role, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        messageElement.classList.add(role);
        messageElement.innerHTML = message;  // Use innerHTML to insert formatted message
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
});
