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

document.getElementById('health-form').addEventListener('submit', function(e) {
  e.preventDefault();
  
  const formData = new FormData(this);
  const loaderContainer = document.getElementById('loader-container');
  
  // Show loading animation
  loaderContainer.style.display = 'flex';
  
  fetch('/analyze', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      document.getElementById('response').innerHTML = `<p class="error">${data.error}</p>`;
    } else {
      document.getElementById('response').innerHTML = data.analysis;
    }
    document.getElementById('result-section').style.display = 'block';
  })
  .catch(error => {
    console.error('Error:', error);
    document.getElementById('response').innerHTML = '<p class="error">An error occurred. Please try again.</p>';
  })
  .finally(() => {
    // Hide loading animation
    loaderContainer.style.display = 'none';
  });
});
