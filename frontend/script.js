document.addEventListener("DOMContentLoaded", function () {
  // Get DOM elements
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("fileElem");
  const fileSelect = document.getElementById("fileSelect");
  const fileName = document.getElementById("file-name");
  const previewImg = document.getElementById("preview");
  const uploadInstructions = document.querySelector(".upload-instructions");
  const removeImageButton = document.getElementById("removeImage");
  const uploadImageButton = document.getElementById("uploadImage");
  // Bind upload button click to upload handler
  uploadImageButton.addEventListener("click", uploadImage);

  // Open file selector when 'Select Image' button is clicked
  fileSelect.addEventListener("click", () => fileInput.click());

  // Handle file selection
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      fileName.textContent = `Selected: ${file.name}`;
      showPreview(file);
    }
  });

  // Handle pasted images from clipboard
  document.addEventListener("paste", (e) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.indexOf("image") !== -1) {
        const file = item.getAsFile();
        fileInput.files = new DataTransfer().files; // Reset file input
        showPreview(file);
        fileName.textContent = `Pasted Image`;
        break;
      }
    }
  });

  // Highlight drop area on drag
  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("highlight");
  });

  // Remove highlight when dragging leaves the area
  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("highlight");
  });

  // Handle dropped files
  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("highlight");

    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      fileName.textContent = `Dropped: ${file.name}`;
      showPreview(file);
    }
  });

  // Remove image and reset UI
  removeImageButton.addEventListener("click", () => {
    fileInput.value = "";
    fileName.textContent = "";
    previewImg.src = "";
    previewImg.style.display = "none";
    uploadInstructions.style.display = "flex";
    dropArea.classList.remove("filled");
    removeImageButton.style.display = "none";
    uploadImageButton.style.display = "none";
  });

  // Upload image to backend and handle response
  async function uploadImage() {
    const input = document.getElementById("fileElem");
    const file = input.files[0];

    const formData = new FormData();
    formData.append("image", file);

    // Send image to backend API
    const response = await fetch(
      "https://where-am-ai-backend.onrender.com/upload",
      {
        method: "POST",
        body: formData,
      }
    );

    const result = await response.json();
    const message = result.message;

    // Extract latitude and longitude from message
    const matches = message.match(/[-+]?[0-9]*\.?[0-9]+/g);
    const latitude = parseFloat(matches[0]);
    const longitude = parseFloat(matches[1]);

    // Show results on map
    showResultPopup(latitude, longitude);
  }

  // Show result popup with map and location details
  function showResultPopup(lat, lon) {
    const popup = document.getElementById("resultPopup");
    popup.style.display = "flex";

    // Remove previous map instance if exists
    if (window.existingMap) {
      window.existingMap.remove();
      document.getElementById("map").innerHTML = "";
    }
    // Initialize new map
    window.existingMap = L.map("map").setView([lat, lon], 14);

    // Add tile layer
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(window.existingMap);

    // Add marker with popup
    L.marker([lat, lon])
      .addTo(window.existingMap)
      .bindPopup("AI guessed here!")
      .openPopup();

    // Use reverse geocoding API to get location name
    fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`
    )
      .then((res) => res.json())
      .then((data) => {
        const city =
          data.address.city ||
          data.address.town ||
          data.address.village ||
          "Unknown City";
        const state = data.address.state || "Unknown State";
        const country = data.address.country || "Unknown Country";

        // Display location info and feedback UI
        document.getElementById("info").innerHTML = `
        <div class="location-info">
            <strong>City:</strong> ${city}<br>
            <strong>State:</strong> ${state}<br>
            <strong>Country:</strong> ${country}
        </div>
        <div class="feedback-section">
        <span class="feedback-text">Was this correct?</span>
        <div class="feedback-buttons">
            <i class="material-icons feedback-icon up">thumb_up</i>
            <i class="material-icons feedback-icon down">thumb_down</i>
        </div>
        </div>
        `;
        setupFeedback();
      })

      .catch((err) => {
        document.getElementById("info").innerHTML = `
        <div class="error-message">
        Failed to fetch location info:<br>
        </div>
        <div class="feedback-section">
          <span class="feedback-text">Was this correct?</span>
          <div class="feedback-buttons">
            <i class="material-icons feedback-icon up">thumb_up</i>
            <i class="material-icons feedback-icon down">thumb_down</i>
          </div>
        </div>
      `;
      
      setupFeedback();
      });
  }

  // Set up feedback button interactions
  function setupFeedback() {
    const upButton = document.querySelector(".feedback-icon.up");
    const downButton = document.querySelector(".feedback-icon.down");

    // Clear previous handlers to prevent duplicates
    upButton.onclick = null;
    downButton.onclick = null;

    // Like button behavior
    upButton.addEventListener("click", function () {
      this.classList.add("active");
      downButton.classList.remove("active");
      console.log("User liked the prediction");
    });

    // Dislike button behavior
    downButton.addEventListener("click", function () {
      this.classList.add("active");
      upButton.classList.remove("active");
      console.log("User disliked the prediction");
    });
  }

  // Display image preview in UI
  function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewImg.style.display = "block";
      uploadInstructions.style.display = "none";
      dropArea.classList.add("filled");
      removeImageButton.style.display = "block";
      uploadImageButton.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
});
// Close the result popup and clear map
function closePopup() {
  document.getElementById("resultPopup").style.display = "none";
  document.getElementById("map").innerHTML = "";

  // Clear the uploaded image and reset UI state
  const fileInput = document.getElementById("fileElem");
  const fileName = document.getElementById("file-name");
  const previewImg = document.getElementById("preview");
  const uploadInstructions = document.querySelector(".upload-instructions");
  const dropArea = document.getElementById("drop-area");
  const removeImageButton = document.getElementById("removeImage");
  const uploadImageButton = document.getElementById("uploadImage");

  fileInput.value = "";
  fileName.textContent = "";
  previewImg.src = "";
  previewImg.style.display = "none";
  uploadInstructions.style.display = "flex";
  dropArea.classList.remove("filled");
  removeImageButton.style.display = "none";
  uploadImageButton.style.display = "none";
}
