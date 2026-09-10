const API_BASE = "/api/v1";

async function loadResult() {
  const root = document.getElementById("result-root");
  const documentName = root.dataset.documentName;
  
  try {
    const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(documentName)}`);
    if (!res.ok) {
      throw new Error(`Document '${documentName}' not found`);
    }
    const data = await res.json();

    renderSummary(data);
    renderFields(data.extracted_data || {});
    renderValidation(data.validation || {});
    renderRawJson(data);

  } catch (err) {
    console.error(err);
    document.getElementById("doc-title").textContent = "Error Loading Document";
    document.getElementById("meta-name").textContent = err.message;
  }
}

function renderSummary(data) {
  const status = data.processing_status || "UNKNOWN";
  const statusClass = status.toLowerCase() === "pass" ? "badge-pass" : "badge-fail";

  const statusBadge = document.getElementById("doc-status-badge");
  statusBadge.className = `badge ${statusClass}`;
  statusBadge.textContent = status;

  document.getElementById("meta-name").textContent = data.document_name || "-";
  document.getElementById("meta-type").textContent = data.document_type || "-";
  
  const ocrUsed = data.processing_metadata?.ocr_used;
  document.getElementById("meta-ocr").innerHTML = ocrUsed 
    ? '<span class="badge badge-na">Tesseract OCR</span>' 
    : '<span class="badge badge-pass">Native Text</span>';

  document.getElementById("meta-time").textContent = `${data.processing_metadata?.processing_time_ms ?? 0} ms`;

  const pageCount = data.file_validation?.page_count ?? 1;
  const fileType = data.file_validation?.file_type || "document";
  document.getElementById("meta-pages").textContent = `${pageCount} page(s) (${fileType})`;

  // Render Issues Alert Box if there are issues or status is FAILED
  const issuesContainer = document.getElementById("issues-container");
  const issues = data.validation?.issues || [];
  
  if (issues.length > 0 || status === "FAILED") {
    let html = `
      <div class="issues-box">
        <div class="issues-title">⚠️ Validation Alerts / Failures (${issues.length})</div>
        <ul class="issues-list">
    `;
    if (issues.length > 0) {
      issues.forEach((issue) => {
        html += `<li>${escapeHtml(issue)}</li>`;
      });
    } else {
      html += `<li>Document validation failed. Check individual field and calculation rules below.</li>`;
    }
    html += `</ul></div>`;
    issuesContainer.innerHTML = html;
  } else {
    issuesContainer.innerHTML = "";
  }
}

function renderFields(extractedData) {
  const tbody = document.getElementById("fields-tbody");
  tbody.innerHTML = "";

  const lineItems = extractedData.line_items;

  for (const [field, info] of Object.entries(extractedData)) {
    if (field === "line_items") continue;

    const value = info?.value;
    const isMissing = value === null || value === undefined;
    const pageNum = info?.page_number ?? info?.evidence?.page_number ?? "-";
    const sourceSnippet = info?.evidence?.source_text;

    const tr = document.createElement("tr");
    
    // Explicit Visual Highlight for Missing Fields (Required by spec)
    if (isMissing) {
      tr.className = "field-missing-row";
    }

    tr.innerHTML = `
      <td style="font-weight: 500;">${escapeHtml(field)}</td>
      <td>
        ${
          isMissing
            ? '<span class="badge badge-missing">Missing / Null</span>'
            : `<strong style="color: var(--text-main);">${escapeHtml(String(value))}</strong>`
        }
      </td>
      <td>${pageNum}</td>
      <td>
        ${
          sourceSnippet 
            ? `<span class="evidence-tag" title="${escapeHtml(sourceSnippet)}">"${escapeHtml(sourceSnippet)}"</span>`
            : '<span style="color: var(--text-muted); font-size: 0.8rem;">None provided</span>'
        }
      </td>
    `;

    tbody.appendChild(tr);
  }

  // Render Line Items Table if present
  const lineItemsCard = document.getElementById("line-items-card");
  const lineItemsTbody = document.getElementById("line-items-tbody");
  
  if (Array.isArray(lineItems) && lineItems.length > 0) {
    lineItemsCard.style.display = "block";
    lineItemsTbody.innerHTML = "";
    
    lineItems.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(item.description || "-")}</td>
        <td>${item.quantity ?? "-"}</td>
        <td>${item.unit_price != null ? `$${item.unit_price}` : "-"}</td>
        <td style="font-weight: 600;">${item.amount != null ? `$${item.amount}` : "-"}</td>
      `;
      lineItemsTbody.appendChild(tr);
    });
  } else {
    lineItemsCard.style.display = "none";
  }
}

function renderValidation(validation) {
  const overallBadge = document.getElementById("validation-overall-badge");
  const overallStatus = validation.overall_status || "NOT_APPLICABLE";
  
  let overallClass = "badge-na";
  if (overallStatus === "PASS") overallClass = "badge-pass";
  if (overallStatus === "FAIL") overallClass = "badge-fail";

  overallBadge.className = `badge ${overallClass}`;
  overallBadge.textContent = overallStatus;

  const tbody = document.getElementById("validation-tbody");
  tbody.innerHTML = "";

  const checks = validation.checks || [];

  if (checks.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No financial validation rules for this document type.</td></tr>`;
    return;
  }

  checks.forEach((check) => {
    const tr = document.createElement("tr");
    const status = check.status || "NOT_APPLICABLE";

    let statusBadgeClass = "badge-na";
    if (status === "PASS") statusBadgeClass = "badge-pass";
    if (status === "FAIL") statusBadgeClass = "badge-fail";

    // Explicit Visual Highlight for Failed Validations (Required by spec)
    if (status === "FAIL") {
      tr.className = "check-fail-row";
    }

    const calcVal = check.calculated_value != null ? check.calculated_value : "-";
    const repVal = check.reported_value != null ? check.reported_value : "-";
    const varianceVal = check.variance != null ? check.variance : "-";

    tr.innerHTML = `
      <td style="font-weight: 500;">${escapeHtml(check.name)}</td>
      <td style="font-family: monospace; font-size: 0.8rem; color: #a5b4fc;">${escapeHtml(check.formula)}</td>
      <td>${calcVal}</td>
      <td>${repVal}</td>
      <td style="${check.variance && check.variance !== 0 ? 'color: var(--status-fail-text); font-weight: 600;' : ''}">${varianceVal}</td>
      <td><span class="badge ${statusBadgeClass}">${status}</span></td>
    `;

    tbody.appendChild(tr);
  });
}

function renderRawJson(data) {
  const jsonPre = document.getElementById("raw-json");
  const copyBtn = document.getElementById("copy-json-btn");

  const formattedStr = JSON.stringify(data, null, 2);
  jsonPre.textContent = formattedStr;

  copyBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigator.clipboard.writeText(formattedStr).then(() => {
      copyBtn.textContent = "Copied!";
      setTimeout(() => {
        copyBtn.textContent = "Copy JSON";
      }, 1500);
    });
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

loadResult();
