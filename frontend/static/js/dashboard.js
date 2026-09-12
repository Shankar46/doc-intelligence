const API_BASE = "/api/v1";

const dropzone = document.getElementById("file-dropzone");
const fileInput = document.getElementById("file");
const dropzoneText = document.getElementById("dropzone-text");
const statusEl = document.getElementById("upload-status");
const submitBtn = document.getElementById("submit-btn");

// Handle File Drag & Drop
if (dropzone && fileInput) {
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      updateFilePreview();
    }
  });

  fileInput.addEventListener("change", updateFilePreview);
}

function updateFilePreview() {
  if (fileInput.files.length > 0) {
    const file = fileInput.files[0];
    dropzoneText.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  } else {
    dropzoneText.textContent = "Click or drag file here";
  }
}

async function loadDocuments() {
  const tbody = document.getElementById("documents-tbody");
  const emptyState = document.getElementById("empty-state");

  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error("Failed to fetch documents");
    const data = await res.json();

    tbody.innerHTML = "";
    const docs = data.documents || [];

    if (docs.length === 0) {
      emptyState.style.display = "block";
      return;
    }

    emptyState.style.display = "none";

    docs.forEach((doc) => {
      const tr = document.createElement("tr");
      const statusClass = (doc.processing_status || "").toLowerCase() === "pass" ? "badge-pass" : "badge-fail";
      let timeString = doc.processed_at;
      if (timeString && !timeString.endsWith('Z') && !timeString.includes('+')) {
          timeString += 'Z';
      }
      const dateStr = timeString ? new Date(timeString).toLocaleString() : "-";

      tr.innerHTML = `
        <td style="font-weight: 500;">
          <a href="/documents/${encodeURIComponent(doc.document_name)}/view" style="color: var(--text-main); text-decoration: none;">
            📄 ${escapeHtml(doc.document_name)}
          </a>
        </td>
        <td><span class="badge badge-type">${escapeHtml(doc.document_type)}</span></td>
        <td><span class="badge ${statusClass}">${escapeHtml(doc.processing_status)}</span></td>
        <td style="color: var(--text-muted); font-size: 0.85rem;">${dateStr}</td>
        <td>
          <a href="/documents/${encodeURIComponent(doc.document_name)}/view" class="btn-sm" style="text-decoration: none; display: inline-block;">
            View Details
          </a>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error(err);
  }
}

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const selectedType = document.getElementById("document_type").value;
  if (!selectedType) {
    showStatus("Please select the document type before processing.", "error");
    return;
  }

  if (!fileInput.files.length) {
    showStatus("Please select a file to upload.", "error");
    return;
  }

  showStatus("Processing document via AI engine...", "info");
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("document_type", document.getElementById("document_type").value);

  try {
    const res = await fetch(`${API_BASE}/documents/process`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      const errMsg = data.error?.message || "Processing failed.";
      showStatus(`Error: ${errMsg}`, "error");
      return;
    }

    const statusType = data.processing_status === "PASS" ? "success" : "error";
    showStatus(`Processing complete: Status is ${data.processing_status}`, statusType);
    
    // Reset file preview
    fileInput.value = "";
    updateFilePreview();
    
    // Refresh document list
    await loadDocuments();

    // Redirect to detail view after 1 second if successful
    setTimeout(() => {
      window.location.href = `/documents/${encodeURIComponent(data.document_name)}/view`;
    }, 1200);

  } catch (err) {
    showStatus("Network error while processing document.", "error");
    console.error(err);
  } finally {
    submitBtn.disabled = false;
  }
});

document.getElementById("refresh-btn")?.addEventListener("click", loadDocuments);

function showStatus(message, type) {
  statusEl.className = type;
  statusEl.textContent = message;
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

loadDocuments();
