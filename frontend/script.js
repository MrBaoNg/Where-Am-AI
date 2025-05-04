document.addEventListener("DOMContentLoaded", function () {
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("fileElem");
  const fileSelect = document.getElementById("fileSelect");
  const fileName = document.getElementById("file-name");
  const previewImg = document.getElementById("preview");
  const uploadInstructions = document.querySelector(".upload-instructions");
  const removeImageButton = document.getElementById("removeImage");
  const uploadImageButton = document.getElementById("uploadImage");
  uploadImageButton.addEventListener("click", uploadImage);

  fileSelect.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      fileName.textContent = `Selected: ${file.name}`;
      showPreview(file);
    }
  });

  document.addEventListener("paste", (e) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.indexOf("image") !== -1) {
        const file = item.getAsFile();
        fileInput.files = new DataTransfer().files;
        showPreview(file);
        fileName.textContent = `Pasted Image`;
        break;
      }
    }
  });

  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("highlight");
  });

  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("highlight");
  });

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
  removeImageButton.addEventListener("click", () => {
    // Reset the form.
    fileInput.value = "";
    fileName.textContent = "";
    previewImg.src = "";
    previewImg.style.display = "none";
    uploadInstructions.style.display = "flex";
    dropArea.classList.remove("filled");
    removeImageButton.style.display = "none";
    uploadImageButton.style.display = "none";
  });
  async function uploadImage() {
    const input = document.getElementById("fileElem");
    const file = input.files[0];

    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch(
      "https://where-am-ai-backend.onrender.com/upload",
      {
        method: "POST",
        body: formData,
      }
    );

    const result = await response.json();
    const message = result.message;

    // Extract lat/lon
    const matches = message.match(/[-+]?[0-9]*\.?[0-9]+/g);
    const latitude = parseFloat(matches[0]);
    const longitude = parseFloat(matches[1]);

    showResultPopup(30.31304, -95.458138);
    // showResultPopup(latitude, longitude);
  }
  function showResultPopup(lat, lon) {
    const popup = document.getElementById("resultPopup");
    popup.style.display = "flex";

    // Clear existing map if it exists
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

    // Add marker
    L.marker([lat, lon])
      .addTo(window.existingMap)
      .bindPopup("AI guessed here!")
      .openPopup();

    // Fetch location info
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
        document.getElementById("info").innerHTML = `
            <strong>City:</strong> ${city}<br>
            <strong>State:</strong> ${state}<br>
            <strong>Country:</strong> ${country}
            `;
      })
      .catch((err) => {
        document.getElementById("info").innerText =
          "Failed to fetch location info.";
      });
  }

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
function closePopup() {
  document.getElementById("resultPopup").style.display = "none";
  document.getElementById("map").innerHTML = "";
}
