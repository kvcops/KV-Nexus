document.addEventListener("DOMContentLoaded", function () {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("user_input");
    const sendButton = document.getElementById("send_button");

    function appendMessage(role, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        messageElement.classList.add(role);
        messageElement.innerHTML = message;
        messagesContainer.appendChild(messageElement);
        
        // Trigger reflow
        messageElement.offsetHeight;
        
        messageElement.classList.add("show");
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function addTypingIndicator() {
        const indicator = document.createElement("div");
        indicator.classList.add("typing-indicator");
        indicator.innerHTML = '<span></span><span></span><span></span>';
        messagesContainer.appendChild(indicator);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function removeTypingIndicator() {
        const indicator = messagesContainer.querySelector(".typing-indicator");
        if (indicator) {
            indicator.remove();
        }
    }

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    function sendMessage() {
        const userMessage = userInput.value;
        if (userMessage.trim() !== "") {
            appendMessage("user", userMessage);
            userInput.value = "";
            
            addTypingIndicator();

            fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message: userMessage }),
            })
                .then(response => response.json())
                .then(data => {
                    removeTypingIndicator();
                    const formattedResponse = formatResponse(data.reply);
                    appendMessage("bot", formattedResponse);
                })
                .catch(error => {
                    console.error('Error fetching response:', error);
                    removeTypingIndicator();
                    appendMessage("bot", 'Error generating response. Please try again later.');
                });
        }
    }

    function formatResponse(response) {
        return response.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/(\d+\.\s+)/g, '<br>$1')
            .replace(/\n/g, '<br>');
    }
});
document.addEventListener("DOMContentLoaded", function () {
    const messagesContainer = document.getElementById("messages");
    const userInput = document.getElementById("user_input");
    const sendButton = document.getElementById("send_button");
    const themeSwitch = document.getElementById("theme-switch");

    function appendMessage(role, message) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message");
        messageElement.classList.add(role);
        messageElement.innerHTML = message;
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function sendMessage() {
        const userMessage = userInput.value;
        if (userMessage.trim() !== "") {
            appendMessage("user", userMessage);
            userInput.value = "";
            
            // Simulated bot response (replace with actual API call)
            setTimeout(() => {
                const botResponse = "Hmm...lets try other queries";
                appendMessage("bot", botResponse);
            }, 1000);
        }
    }

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    // Theme toggling
    themeSwitch.addEventListener("change", function() {
        document.body.classList.toggle("dark-theme");
    });

    // Particle background effect
    function createParticle() {
        const particle = document.createElement("div");
        particle.classList.add("particle");
        particle.style.left = Math.random() * 100 + "vw";
        particle.style.animationDuration = Math.random() * 3 + 2 + "s";
        document.body.appendChild(particle);

        setTimeout(() => {
            particle.remove();
        }, 5000);
    }

    setInterval(createParticle, 200);
});

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Call this function after appending a new message
function appendMessage(role, message) {
    // ... existing code ...
    scrollToBottom();
}
function smoothScrollToBottom() {
    const scrollHeight = messagesContainer.scrollHeight;
    const currentScroll = messagesContainer.scrollTop;
    const targetScroll = scrollHeight - messagesContainer.clientHeight;
    
    if (targetScroll > currentScroll) {
        anime({
            targets: messagesContainer,
            scrollTop: targetScroll,
            duration: 500,
            easing: 'easeInOutQuad'
        });
    }
}

// Use this instead of the previous scrollToBottom function
