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
        responseElement.innerHTML = `<p>${result.analysis}</p>`;

        // Update the form with additional inputs for intermediate and final analysis
        if (result.stage === 'initial') {
            addInputField('additionalSymptoms', 'Enter additional symptoms based on initial analysis');
        } else if (result.stage === 'intermediate') {
            addInputField('finalSymptoms', 'Enter final symptoms based on intermediate analysis');
        }
    } catch (error) {
        responseElement.innerHTML = `<p>Error: ${error.message}</p>`;
    }
});

function addInputField(name, placeholder) {
    const newInput = document.createElement('input');
    newInput.type = 'text';
    newInput.name = name;
    newInput.placeholder = placeholder;
    document.getElementById('chatbot-form').appendChild(newInput);
}
