document.getElementById('chatbot-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const responseElement = document.getElementById('response');
    responseElement.innerHTML = 'Processing...';

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // --- Format the response ---
         // --- Format the response (updated) ---
         const analysisLines = result.analysis.split("\n"); // Split by newline
         const listItems = analysisLines.map(line => {
             if (line.trim().startsWith("* ")) { 
                 return `<li>${line.replace("* ", "")}</li>`;
             } else {
                 return `<p>${line}</p>`; 
             }
         });
         responseElement.innerHTML = `<ul>${listItems.join("")}</ul>`;
        // --- End of formatting ---

    } catch (error) {
        responseElement.innerHTML = `<p>Error: ${error.message}</p>`;
    }
});
