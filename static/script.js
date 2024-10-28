function handleImageUpload() {
  const imageInput = document.getElementById('image');
  const previewSection = document.getElementById('preview-section');
  const imagePreview = document.getElementById('image-preview');

  if (imageInput.files && imageInput.files[0]) {
    const reader = new FileReader();
    reader.onload = function(e) {
      imagePreview.src = e.target.result;
      previewSection.style.display = 'block';
    };
    reader.readAsDataURL(imageInput.files[0]);
  } else {
    previewSection.style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', function() {
  const messagesContainer = document.getElementById('messages-container');
  const userInput = document.getElementById('user-input');
  const sendButton = document.getElementById('send-button');
  const themeToggleButton = document.getElementById('theme-toggle-button');

  // Consolidated function to append messages
  function appendMessage(content, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.textContent = content;
    messagesContainer.appendChild(messageElement);
    smoothScrollToBottom();
  }

  // Smooth scrolling function
  function smoothScrollToBottom() {
    messagesContainer.scrollTo({
      top: messagesContainer.scrollHeight,
      behavior: 'smooth'
    });
  }

  // Event handler for sending a message
  function handleSendMessage() {
    const message = userInput.value.trim();
    if (message) {
      appendMessage(message, 'user');
      userInput.value = ''; // Clear input after sending
      // Logic for bot response goes here, e.g., appendMessage("Hello, I'm a bot!", 'bot');
    }
  }

  // Event listener for send button
  sendButton.addEventListener('click', handleSendMessage);

  // Event listener for Enter key to send message
  userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
      handleSendMessage();
    }
  });

  // Theme toggle functionality
  themeToggleButton.addEventListener('click', function() {
    document.body.classList.toggle('dark-theme');
  });
});
