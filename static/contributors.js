// Initialize AOS Library
AOS.init({
    duration: 1000,
    once: true,
  });
  
  // GitHub API URLs
  const contributorsApiUrl = 'https://api.github.com/repos/kvcops/KV-Nexus/contributors?per_page=100';
  const repoApiUrl = 'https://api.github.com/repos/kvcops/KV-Nexus';
  
  // Function to fetch contributors data
  async function fetchContributors() {
    try {
      const response = await fetch(contributorsApiUrl);
      const contributors = await response.json();
      return contributors;
    } catch (error) {
      console.error('Error fetching contributors:', error);
    }
  }
  
  // Function to fetch repository data
  async function fetchRepoData() {
    try {
      const response = await fetch(repoApiUrl);
      const repoData = await response.json();
      return repoData;
    } catch (error) {
      console.error('Error fetching repository data:', error);
    }
  }
  
  // Function to create contributor cards
  function createContributorCards(contributors) {
    const grid = document.getElementById('contributors-grid');
  
    contributors.forEach((contributor, index) => {
      // Create card elements
      const card = document.createElement('div');
      card.classList.add('contributor-card');
      card.setAttribute('data-aos', 'fade-up');
      card.setAttribute('data-aos-delay', `${index * 100}`);
  
      const cardInner = document.createElement('div');
      cardInner.classList.add('card-inner');
  
      const cardFront = document.createElement('div');
      cardFront.classList.add('card-front');
  
      const avatar = document.createElement('img');
      avatar.src = contributor.avatar_url;
      avatar.alt = `Photo of ${contributor.login}`;
  
      const name = document.createElement('h2');
      name.textContent = contributor.login;
  
      const contributions = document.createElement('p');
      contributions.textContent = `Contributions: ${contributor.contributions}`;
  
      cardFront.appendChild(avatar);
      cardFront.appendChild(name);
      cardFront.appendChild(contributions);
  
      const cardBack = document.createElement('div');
      cardBack.classList.add('card-back');
  
      const backName = document.createElement('h2');
      backName.textContent = contributor.login;
  
      const bio = document.createElement('p');
      bio.textContent = 'Fetching bio...'; // Placeholder text
  
      const socialLinks = document.createElement('div');
      socialLinks.classList.add('social-links');
  
      const githubLink = document.createElement('a');
      githubLink.href = contributor.html_url;
      githubLink.target = '_blank';
      githubLink.innerHTML = '<i class="fab fa-github"></i>';
  
      socialLinks.appendChild(githubLink);
  
      cardBack.appendChild(backName);
      cardBack.appendChild(bio);
      cardBack.appendChild(socialLinks);
  
      cardInner.appendChild(cardFront);
      cardInner.appendChild(cardBack);
      card.appendChild(cardInner);
      grid.appendChild(card);
  
      // Fetch additional user data (like bio)
      fetchUserData(contributor.login, bio);
  
      // Add click event listener for flipping on touch devices
      card.addEventListener('click', () => {
        card.classList.toggle('is-flipped');
      });
    });
  }
  
  // Function to fetch individual user data
  async function fetchUserData(username, bioElement) {
    const userApiUrl = `https://api.github.com/users/${username}`;
    try {
      const response = await fetch(userApiUrl);
      const userData = await response.json();
      bioElement.textContent = userData.bio || 'No bio available.';
    } catch (error) {
      console.error(`Error fetching user data for ${username}:`, error);
      bioElement.textContent = 'No bio available.';
    }
  }
  
  // Main function to initiate everything
  async function init() {
    const contributors = await fetchContributors();
    if (contributors) {
      createContributorCards(contributors);
      AOS.refresh(); // Refresh AOS to animate newly added elements
    }
  }
  
  init();