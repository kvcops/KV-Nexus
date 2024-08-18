function handleImageUpload() {
    const image = document.getElementById('image').files[0];
    const imagePreview = document.getElementById('image-preview');
    const previewSection = document.getElementById('preview-section');
    
    if (image) {
      const reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        previewSection.style.display = 'flex';
      };
      reader.readAsDataURL(image);
    } else {
      imagePreview.src = "";
      previewSection.style.display = 'none';
    }
  }
  
  document.getElementById('health-form').addEventListener('submit', async function(event) {
    event.preventDefault();
  
    const formData = new FormData(event.target);
    const responseElement = document.getElementById('response');
    responseElement.innerHTML = '<p class="loading">Analyzing your health information...</p>';
  
    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData
      });
  
      const result = await response.json();
  
      const analysisLines = result.analysis.split("\n");
      const formattedResponse = analysisLines.map(line => {
        if (line.trim().startsWith("* ")) {
          return `<li>${line.replace("* ", "")}</li>`;
        } else {
          return `<p>${line}</p>`;
        }
      }).join("");
  
      responseElement.innerHTML = `<ul>${formattedResponse}</ul>`;
      document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
      responseElement.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
  });
