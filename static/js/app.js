/**
 * CivicFind Frontend Client Application
 * Information Retrieval Lab System
 */

// Application State
const state = {
  currentQuery: "",
  activeTab: "search",
  selectedModel: "bm25",
  useExpansion: false,
  activeCategory: "All",
  activeLevel: "All",
  searchResults: [],
  relevantDocIds: new Set(),
  nonRelevantDocIds: new Set(),
  catalogPage: 1,
  catalogTotalPages: 1,
  chartInstance: null
};

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  loadIndexStats();
  loadCatalogCategories();
  loadCatalog(1);
  runPlaygroundPreprocessing();
  inspectTermPostings("scholarship");
  // Pre-load evaluation in background so charts are ready
  runSystemEvaluation(false);
});

function initEventListeners() {
  const queryInput = document.getElementById("searchQueryInput");
  const clearBtn = document.getElementById("clearSearchBtn");

  queryInput.addEventListener("input", (e) => {
    if (e.target.value.trim().length > 0) {
      clearBtn.classList.remove("hidden");
    } else {
      clearBtn.classList.add("hidden");
      document.getElementById("autocompleteDropdown").classList.add("hidden");
    }
  });

  // Close autocomplete when clicking outside
  document.addEventListener("click", (e) => {
    const dropdown = document.getElementById("autocompleteDropdown");
    if (!dropdown.contains(e.target) && e.target !== queryInput) {
      dropdown.classList.add("hidden");
    }
  });
}

// ----------------- TAB NAVIGATION -----------------
function switchTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab-view").forEach(el => el.classList.remove("active-view"));
  document.querySelectorAll(".nav-tab").forEach(el => el.classList.remove("active-tab"));

  const targetView = document.getElementById(`view-${tabName}`);
  const targetBtn = document.getElementById(`tab-btn-${tabName}`);

  if (targetView) targetView.classList.add("active-view");
  if (targetBtn) targetBtn.classList.add("active-tab");

  window.scrollTo({ top: 0, behavior: "smooth" });

  if (tabName === "evaluation" && !state.chartInstance) {
    runSystemEvaluation(false);
  }
}

// ----------------- SEARCH & AUTOCOMPLETE -----------------
let autocompleteDebounceTimer = null;
function handleAutocomplete(val) {
  clearTimeout(autocompleteDebounceTimer);
  const dropdown = document.getElementById("autocompleteDropdown");
  const cleanVal = val.trim().toLowerCase();

  if (cleanVal.length < 2) {
    dropdown.classList.add("hidden");
    return;
  }

  autocompleteDebounceTimer = setTimeout(async () => {
    try {
      const res = await fetch(`/api/suggest?q=${encodeURIComponent(cleanVal)}`);
      const suggestions = await res.json();

      if (!suggestions || suggestions.length === 0) {
        dropdown.classList.add("hidden");
        return;
      }

      dropdown.innerHTML = suggestions.map(s => `
        <div class="px-4 py-2 hover:bg-sky-50 cursor-pointer text-xs text-slate-700 flex items-center justify-between" onclick="selectSuggestion('${s}')">
          <span><i class="fa-solid fa-magnifying-glass text-slate-400 mr-2 text-[10px]"></i>${s}</span>
          <span class="text-[10px] text-slate-400">vocabulary term</span>
        </div>
      `).join("");
      dropdown.classList.remove("hidden");
    } catch (err) {
      console.error("Autocomplete fetch error:", err);
    }
  }, 150);
}

function selectSuggestion(term) {
  const queryInput = document.getElementById("searchQueryInput");
  queryInput.value = term;
  document.getElementById("autocompleteDropdown").classList.add("hidden");
  handleSearch(new Event("submit"));
}

function clearSearch() {
  document.getElementById("searchQueryInput").value = "";
  document.getElementById("clearSearchBtn").classList.add("hidden");
  document.getElementById("autocompleteDropdown").classList.add("hidden");
  document.getElementById("searchQueryInput").focus();
}

function triggerSampleSearch(sampleQuery) {
  switchTab("search");
  document.getElementById("searchQueryInput").value = sampleQuery;
  document.getElementById("clearSearchBtn").classList.remove("hidden");
  handleSearch(new Event("submit"));
}

function handleModelChange() {
  state.selectedModel = document.getElementById("modelSelector").value;
  if (state.currentQuery) {
    handleSearch(new Event("submit"));
  }
}

function handleFilterChange() {
  state.activeCategory = document.getElementById("categoryFilter").value;
  state.activeLevel = document.getElementById("levelFilter").value;
  if (state.currentQuery) {
    handleSearch(new Event("submit"));
  }
}

async function handleSearch(e) {
  if (e && e.preventDefault) e.preventDefault();
  const queryInput = document.getElementById("searchQueryInput");
  const query = queryInput.value.trim();

  if (!query) return;

  state.currentQuery = query;
  state.selectedModel = document.getElementById("modelSelector").value;
  state.useExpansion = document.getElementById("queryExpansionCheckbox").checked;
  state.activeCategory = document.getElementById("categoryFilter").value;
  state.activeLevel = document.getElementById("levelFilter").value;

  // Reset feedback state on new fresh query
  state.relevantDocIds.clear();
  state.nonRelevantDocIds.clear();
  updateFeedbackBanner();

  const container = document.getElementById("searchResultsContainer");
  container.innerHTML = `
    <div class="bg-white rounded-xl p-8 border border-slate-200 text-center space-y-3 shadow-sm">
      <div class="inline-block animate-spin text-sky-600 text-2xl"><i class="fa-solid fa-circle-notch"></i></div>
      <p class="text-xs font-semibold text-slate-700">Calculating term frequencies, IDF weights, and document relevance...</p>
    </div>
  `;

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        model: state.selectedModel,
        use_expansion: state.useExpansion,
        filters: {
          category: state.activeCategory,
          level: state.activeLevel
        },
        top_k: 25
      })
    });

    const data = await res.json();
    state.searchResults = data.results || [];
    renderSearchResults(data);

    // Update playground input with search query
    document.getElementById("playgroundInput").value = query;
    runPlaygroundPreprocessing();

  } catch (err) {
    console.error("Search API error:", err);
    container.innerHTML = `
      <div class="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-4 text-xs">
        Failed to execute retrieval query: ${err.message}
      </div>
    `;
  }
}

// ----------------- RENDER RESULTS -----------------
function renderSearchResults(data) {
  const container = document.getElementById("searchResultsContainer");
  const metaBar = document.getElementById("searchMetaBar");
  const spellingBanner = document.getElementById("spellingBanner");
  const results = data.results || [];
  const qInfo = data.query_info || {};

  // Spelling Suggestion
  if (qInfo.spelling_suggestion) {
    document.getElementById("spellingSuggestionText").innerText = qInfo.spelling_suggestion.suggested;
    spellingBanner.classList.remove("hidden");
  } else {
    spellingBanner.classList.add("hidden");
  }

  // Meta Diagnostics Bar
  metaBar.classList.remove("hidden");
  document.getElementById("resultsCountSpan").innerText = `Found ${qInfo.total_matches} schemes`;
  document.getElementById("execTimeSpan").innerHTML = `<i class="fa-solid fa-bolt text-amber-500 mr-1"></i>${qInfo.execution_time_ms} ms`;
  document.getElementById("modelBadgeSpan").innerText = qInfo.model_used;

  // Expansion terms
  const expandedSpan = document.getElementById("expandedTermsSpan");
  if (qInfo.expansion && qInfo.expansion.expanded_terms && qInfo.expansion.expanded_terms.length > 0) {
    expandedSpan.classList.remove("hidden");
    document.getElementById("expandedTermsList").innerText = qInfo.expansion.expanded_terms.slice(0, 6).join(", ");
  } else {
    expandedSpan.classList.add("hidden");
  }

  if (results.length === 0) {
    container.innerHTML = `
      <div class="bg-white rounded-xl p-10 border border-slate-200 text-center space-y-3 shadow-sm">
        <div class="text-3xl text-slate-300"><i class="fa-solid fa-folder-open"></i></div>
        <h4 class="font-bold text-slate-800 text-sm">No schemes matched your search criteria</h4>
        <p class="text-xs text-slate-500 max-w-sm mx-auto">
          Try broader keywords, enable <strong>Query Expansion</strong>, or switch model to <strong>Latent Semantic Indexing</strong>.
        </p>
      </div>
    `;
    return;
  }

  container.innerHTML = results.map((item, index) => {
    const sch = item.scheme;
    const rank = index + 1;
    const relPct = item.relevance_pct || 0;
    const docId = item.doc_id;

    // Determine relevance badge color
    let badgeClass = "rel-fair";
    if (relPct >= 75) badgeClass = "rel-high";
    else if (relPct >= 40) badgeClass = "rel-medium";

    const isRel = state.relevantDocIds.has(docId);
    const isNonRel = state.nonRelevantDocIds.has(docId);

    return `
      <div class="result-card bg-white rounded-xl p-5 border border-slate-200 shadow-sm relative overflow-hidden" id="card-${docId}">
        <div class="flex flex-col sm:flex-row items-start justify-between gap-3">
          
          <div class="space-y-2 flex-grow">
            <!-- Badges & Rank -->
            <div class="flex items-center flex-wrap gap-1.5 text-xs">
              <span class="font-bold text-slate-400 font-mono text-[11px]">#${rank}</span>
              <span class="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-sky-100 text-sky-800">
                ${sch.category || "General"}
              </span>
              <span class="px-2 py-0.5 rounded-full text-[10px] font-medium bg-slate-100 text-slate-700">
                <i class="fa-solid fa-location-dot text-[9px] mr-1 text-slate-400"></i>${sch.level || "Central"}
              </span>
              <span class="text-[10px] text-slate-400 font-mono">${docId}</span>
            </div>

            <!-- Title -->
            <h3 class="text-base sm:text-lg font-bold text-slate-900 hover:text-sky-600 transition cursor-pointer" onclick="openSchemeModal('${docId}')">
              ${sch.name}
            </h3>

            <!-- Department -->
            <p class="text-[11px] text-slate-500">
              <i class="fa-solid fa-building-columns mr-1 text-slate-400"></i>${sch.department || "Government Department"}
            </p>

            <!-- Target Beneficiaries & Eligibility Snippet -->
            <div class="text-xs text-slate-600 space-y-1 pt-1">
              <p><strong class="text-slate-700">Beneficiaries:</strong> ${sch.target_beneficiaries || "Eligible citizens"}</p>
              <p class="line-clamp-2"><strong class="text-slate-700">Eligibility:</strong> ${sch.eligibility || "See details"}</p>
            </div>

            <!-- Matching Terms Highlight -->
            ${item.matching_terms && item.matching_terms.length > 0 ? `
              <div class="flex items-center flex-wrap gap-1 pt-1 text-[11px]">
                <span class="text-slate-400 text-[10px]">Matched:</span>
                ${item.matching_terms.map(t => `<span class="bg-amber-100 text-amber-900 px-1.5 py-0.2 rounded font-mono text-[10px]">${t}</span>`).join("")}
              </div>
            ` : ''}

          </div>

          <!-- Right Column: Relevance Metric Badge & Actions -->
          <div class="flex flex-col sm:items-end justify-between self-stretch shrink-0 space-y-3 pt-1">
            
            <!-- Relevance Gauge Badge -->
            <div class="border rounded-xl px-3 py-1.5 text-center ${badgeClass} shadow-xs">
              <div class="text-[10px] uppercase font-bold tracking-wider">Relevance</div>
              <div class="text-lg font-extrabold leading-none mt-0.5">${relPct}%</div>
            </div>

            <!-- Rocchio Feedback Interactive Thumb Buttons -->
            <div class="flex items-center space-x-1.5 pt-1 text-xs">
              <span class="text-[10px] text-slate-400 mr-1" title="Train Rocchio Relevance Feedback vector">Feedback:</span>
              <button 
                onclick="markFeedback('${docId}', true)" 
                class="w-7 h-7 rounded-lg border flex items-center justify-center transition ${isRel ? 'bg-emerald-600 text-white border-emerald-600 shadow' : 'bg-slate-50 hover:bg-emerald-50 text-slate-500 hover:text-emerald-700 border-slate-200'}"
                title="Mark this document as Relevant for Rocchio feedback"
              >
                <i class="fa-solid fa-thumbs-up text-xs"></i>
              </button>
              <button 
                onclick="markFeedback('${docId}', false)" 
                class="w-7 h-7 rounded-lg border flex items-center justify-center transition ${isNonRel ? 'bg-rose-600 text-white border-rose-600 shadow' : 'bg-slate-50 hover:bg-rose-50 text-slate-500 hover:text-rose-700 border-slate-200'}"
                title="Mark this document as Non-Relevant for Rocchio feedback"
              >
                <i class="fa-solid fa-thumbs-down text-xs"></i>
              </button>
            </div>

            <!-- Action Buttons -->
            <div class="flex items-center space-x-2 pt-2">
              <button onclick="openSchemeModal('${docId}')" class="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-semibold transition">
                Details
              </button>
              ${sch.official_portal ? `
                <a href="${sch.official_portal}" target="_blank" rel="noopener noreferrer" class="px-3 py-1.5 bg-sky-50 hover:bg-sky-100 text-sky-700 border border-sky-200 rounded-lg text-xs font-semibold transition flex items-center space-x-1">
                  <span>Portal</span>
                  <i class="fa-solid fa-arrow-up-right-from-square text-[9px]"></i>
                </a>
              ` : ''}
            </div>

          </div>

        </div>
      </div>
    `;
  }).join("");
}

// ----------------- ROCCHIO RELEVANCE FEEDBACK -----------------
function markFeedback(docId, isRelevant) {
  if (isRelevant) {
    if (state.relevantDocIds.has(docId)) {
      state.relevantDocIds.delete(docId);
    } else {
      state.relevantDocIds.add(docId);
      state.nonRelevantDocIds.delete(docId);
    }
  } else {
    if (state.nonRelevantDocIds.has(docId)) {
      state.nonRelevantDocIds.delete(docId);
    } else {
      state.nonRelevantDocIds.add(docId);
      state.relevantDocIds.delete(docId);
    }
  }
  updateFeedbackBanner();
  // Re-render result cards to reflect thumb states
  renderSearchResults({
    results: state.searchResults,
    query_info: {
      query: state.currentQuery,
      total_matches: state.searchResults.length,
      execution_time_ms: 0,
      model_used: state.selectedModel
    }
  });
}

function updateFeedbackBanner() {
  const banner = document.getElementById("rocchioFeedbackBanner");
  const relCount = state.relevantDocIds.size;
  const nrelCount = state.nonRelevantDocIds.size;

  document.getElementById("relevantCountSpan").innerText = relCount;
  document.getElementById("nonRelevantCountSpan").innerText = nrelCount;

  if (relCount > 0 || nrelCount > 0) {
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

function resetFeedbackSelection() {
  state.relevantDocIds.clear();
  state.nonRelevantDocIds.clear();
  updateFeedbackBanner();
  renderSearchResults({
    results: state.searchResults,
    query_info: {
      query: state.currentQuery,
      total_matches: state.searchResults.length,
      execution_time_ms: 0,
      model_used: state.selectedModel
    }
  });
}

async function applyRocchioRelevanceFeedback() {
  if (!state.currentQuery || (state.relevantDocIds.size === 0 && state.nonRelevantDocIds.size === 0)) {
    return;
  }

  const container = document.getElementById("searchResultsContainer");
  container.innerHTML = `
    <div class="bg-white rounded-xl p-8 border border-slate-200 text-center space-y-3 shadow-sm">
      <div class="inline-block animate-spin text-emerald-600 text-2xl"><i class="fa-solid fa-arrows-spin"></i></div>
      <p class="text-xs font-semibold text-slate-700">Recalculating Rocchio vector weights and re-ranking documents...</p>
    </div>
  `;

  try {
    const res = await fetch("/api/relevance-feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        original_query: state.currentQuery,
        relevant_doc_ids: Array.from(state.relevantDocIds),
        non_relevant_doc_ids: Array.from(state.nonRelevantDocIds),
        top_k: 25
      })
    });

    const data = await res.json();
    state.searchResults = data.results || [];
    renderSearchResults({
      results: state.searchResults,
      query_info: {
        query: state.currentQuery,
        total_matches: state.searchResults.length,
        execution_time_ms: data.execution_time_ms,
        model_used: "Rocchio Relevance Feedback (Reformulated VSM)",
        expansion: {
          expanded_terms: data.feedback_info ? data.feedback_info.added_terms : []
        }
      }
    });

  } catch (err) {
    console.error("Rocchio feedback error:", err);
  }
}

function applySpellingSuggestion() {
  const suggested = document.getElementById("spellingSuggestionText").innerText;
  document.getElementById("searchQueryInput").value = suggested;
  document.getElementById("spellingBanner").classList.add("hidden");
  handleSearch(new Event("submit"));
}

function dismissSpelling() {
  document.getElementById("spellingBanner").classList.add("hidden");
}

// ----------------- TAB 2: IR LAB INSPECTOR -----------------
async function runPlaygroundPreprocessing() {
  const text = document.getElementById("playgroundInput").value.trim();
  if (!text) return;

  try {
    const res = await fetch("/api/preprocess-trace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });
    const trace = await res.json();

    document.getElementById("pipeLowercased").innerText = trace.lowercased || "none";
    document.getElementById("pipeTokens").innerHTML = (trace.tokens || []).map(t => `<span class="bg-sky-100 text-sky-800 px-1 py-0.5 rounded">${t}</span>`).join(" ") || "none";
    document.getElementById("pipeNoStopwords").innerHTML = (trace.no_stopwords || []).map(t => `<span class="bg-amber-100 text-amber-800 px-1 py-0.5 rounded">${t}</span>`).join(" ") || "none";
    document.getElementById("pipeLemmatized").innerHTML = (trace.lemmatized || []).map(t => `<span class="bg-emerald-100 text-emerald-800 px-1 py-0.5 rounded">${t}</span>`).join(" ") || "none";
    document.getElementById("pipeStemmed").innerHTML = (trace.stemmed || []).map(t => `<span class="bg-purple-100 text-purple-800 px-1 py-0.5 rounded">${t}</span>`).join(" ") || "none";

  } catch (err) {
    console.error("Preprocessing trace error:", err);
  }
}

async function inspectTermPostings(initialTerm) {
  const term = initialTerm || document.getElementById("postingsLookupInput").value.trim();
  if (!term) return;

  try {
    const res = await fetch(`/api/index/term/${encodeURIComponent(term)}`);
    const data = await res.json();

    document.getElementById("termStatsRow").classList.remove("hidden");
    document.getElementById("postingsTableWrapper").classList.remove("hidden");

    document.getElementById("termDfVal").innerText = data.document_frequency || 0;
    document.getElementById("termCfVal").innerText = data.collection_frequency || 0;
    document.getElementById("termIdfVal").innerText = data.idf ? data.idf.toFixed(3) : "0.00";
    document.getElementById("termTotalDocsVal").innerText = data.total_documents || 174;

    const tbody = document.getElementById("postingsTableBody");
    const postings = data.postings || [];

    if (postings.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="px-4 py-4 text-center text-slate-400">Term '${term}' does not appear in collection index</td></tr>`;
      return;
    }

    tbody.innerHTML = postings.map(p => `
      <tr class="hover:bg-slate-50 transition">
        <td class="px-4 py-2 font-mono font-bold text-sky-700 cursor-pointer" onclick="openSchemeModal('${p.doc_id}')">${p.doc_id}</td>
        <td class="px-4 py-2 font-medium text-slate-800">${p.scheme_name}</td>
        <td class="px-4 py-2 text-slate-500">${p.category}</td>
        <td class="px-4 py-2 text-center font-bold text-purple-700">${p.tf}</td>
        <td class="px-4 py-2 font-mono text-[10px] text-slate-600">[${p.positions.join(", ")}]</td>
      </tr>
    `).join("");

  } catch (err) {
    console.error("Term inspection error:", err);
  }
}

async function loadIndexStats() {
  try {
    const res = await fetch("/api/index/stats");
    const stats = await res.json();

    document.getElementById("statTotalDocs").innerText = stats.num_documents || 174;
    document.getElementById("statVocabSize").innerText = (stats.vocabulary_size || 2576).toLocaleString();
    document.getElementById("statTotalTokens").innerText = (stats.total_tokens || 18828).toLocaleString();
    document.getElementById("statAvgDocLen").innerText = `${stats.avg_doc_length} tokens`;
  } catch (err) {
    console.error("Index stats error:", err);
  }
}

// ----------------- TAB 3: IR EVALUATION BENCHMARK -----------------
async function runSystemEvaluation(notify = true) {
  try {
    const res = await fetch("/api/evaluate");
    const evalData = await res.json();

    const summaryList = evalData.models_summary || [];
    const bm25Summary = summaryList.find(m => m.model_name.includes("BM25")) || summaryList[0];

    // Update KPI Cards with BM25 benchmark metrics
    if (bm25Summary) {
      document.getElementById("kpiMapVal").innerText = bm25Summary.map.toFixed(4);
      document.getElementById("kpiNdcgVal").innerText = bm25Summary["mean_ndcg@10"].toFixed(4);
      document.getElementById("kpiP5Val").innerText = bm25Summary["mean_p@5"].toFixed(4);
      document.getElementById("kpiMrrVal").innerText = bm25Summary.mrr.toFixed(4);
    }

    // Render Comparative Table
    const tbody = document.getElementById("modelsSummaryTableBody");
    tbody.innerHTML = summaryList.map((m, idx) => {
      const isLeader = m.model_name.includes("BM25") || m.model_name.includes("Hybrid");
      return `
        <tr class="hover:bg-slate-50 transition ${isLeader ? 'bg-sky-50/30' : ''}">
          <td class="px-4 py-3 font-bold text-slate-800 flex items-center">
            ${isLeader ? '<i class="fa-solid fa-crown text-amber-500 mr-2 text-[10px]"></i>' : ''}
            ${m.model_name}
          </td>
          <td class="px-3 py-3 text-center font-mono">${m["mean_p@1"].toFixed(3)}</td>
          <td class="px-3 py-3 text-center font-mono font-semibold">${m["mean_p@5"].toFixed(3)}</td>
          <td class="px-3 py-3 text-center font-mono">${m["mean_p@10"].toFixed(3)}</td>
          <td class="px-3 py-3 text-center font-mono">${m["mean_recall@5"].toFixed(3)}</td>
          <td class="px-3 py-3 text-center font-mono">${m["mean_f1@5"].toFixed(3)}</td>
          <td class="px-3 py-3 text-center font-mono font-bold text-emerald-600">${m.map.toFixed(4)}</td>
          <td class="px-3 py-3 text-center font-mono font-bold text-sky-600">${m["mean_ndcg@5"].toFixed(4)}</td>
          <td class="px-3 py-3 text-center font-mono font-bold text-sky-700">${m["mean_ndcg@10"].toFixed(4)}</td>
          <td class="px-3 py-3 text-center font-mono">${m.mrr.toFixed(3)}</td>
        </tr>
      `;
    }).join("");

    // Render Per-Query BM25 Breakdown Table
    const perQueryBody = document.getElementById("perQueryTableBody");
    const queries = evalData.bm25_details || [];
    perQueryBody.innerHTML = queries.map(q => `
      <tr class="hover:bg-slate-50 transition">
        <td class="px-3 py-2 text-sky-700 font-bold">${q.query_id}</td>
        <td class="px-4 py-2 font-sans text-slate-800">${q.query}</td>
        <td class="px-3 py-2 text-center text-slate-500 font-sans">${q.category}</td>
        <td class="px-3 py-2 text-center font-bold text-slate-700">${q.num_relevant}</td>
        <td class="px-3 py-2 text-center">${q["p@5"].toFixed(2)}</td>
        <td class="px-3 py-2 text-center">${q["recall@5"].toFixed(2)}</td>
        <td class="px-3 py-2 text-center font-bold text-emerald-600">${q.ap.toFixed(3)}</td>
        <td class="px-3 py-2 text-center font-bold text-sky-600">${q["ndcg@10"].toFixed(3)}</td>
        <td class="px-3 py-2 text-center">${q.rr.toFixed(2)}</td>
      </tr>
    `).join("");

    // Render Chart.js
    renderComparisonChart(summaryList);

  } catch (err) {
    console.error("Evaluation run error:", err);
  }
}

function renderComparisonChart(models) {
  const ctx = document.getElementById("evalChart");
  if (!ctx) return;

  if (state.chartInstance) {
    state.chartInstance.destroy();
  }

  const labels = ["P@1", "P@5", "P@10", "Recall@5", "MAP", "NDCG@5", "NDCG@10", "MRR"];

  const colors = [
    { bg: "rgba(2, 132, 199, 0.8)", border: "#0284c7" },   // BM25
    { bg: "rgba(124, 58, 237, 0.8)", border: "#7c3aed" },   // VSM
    { bg: "rgba(5, 150, 105, 0.8)", border: "#059669" },   // Semantic
    { bg: "rgba(217, 119, 6, 0.8)", border: "#d97706" },    // Hybrid
    { bg: "rgba(100, 116, 139, 0.8)", border: "#64748b" }   // Boolean
  ];

  const datasets = models.map((m, idx) => {
    const c = colors[idx % colors.length];
    return {
      label: m.model_name,
      data: [
        m["mean_p@1"],
        m["mean_p@5"],
        m["mean_p@10"],
        m["mean_recall@5"],
        m.map,
        m["mean_ndcg@5"],
        m["mean_ndcg@10"],
        m.mrr
      ],
      backgroundColor: c.bg,
      borderColor: c.border,
      borderWidth: 1,
      borderRadius: 4
    };
  });

  state.chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          grid: { color: "rgba(226, 232, 240, 0.6)" },
          ticks: { font: { family: "Plus Jakarta Sans", size: 11 } }
        },
        x: {
          grid: { display: false },
          ticks: { font: { family: "Plus Jakarta Sans", size: 11, weight: "600" } }
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: { font: { family: "Plus Jakarta Sans", size: 11 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(4)}`
          }
        }
      }
    }
  });
}

// ----------------- TAB 4: SCHEMES CATALOG -----------------
async function loadCatalogCategories() {
  try {
    const res = await fetch("/api/schemes?page=1&per_page=1");
    const data = await res.json();
    const catSelect = document.getElementById("catalogCategorySelect");
    if (data.categories) {
      data.categories.forEach(cat => {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.innerText = cat;
        catSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("Categories fetch error:", err);
  }
}

let catalogDebounce = null;
function handleCatalogFilter() {
  clearTimeout(catalogDebounce);
  catalogDebounce = setTimeout(() => {
    state.catalogPage = 1;
    loadCatalog(1);
  }, 200);
}

function changeCatalogPage(delta) {
  const targetPage = state.catalogPage + delta;
  if (targetPage >= 1 && targetPage <= state.catalogTotalPages) {
    loadCatalog(targetPage);
  }
}

async function loadCatalog(page) {
  state.catalogPage = page;
  const search = document.getElementById("catalogSearchInput").value.trim();
  const category = document.getElementById("catalogCategorySelect").value;
  const level = document.getElementById("catalogLevelSelect").value;

  try {
    const url = `/api/schemes?page=${page}&per_page=15&search=${encodeURIComponent(search)}&category=${encodeURIComponent(category)}&level=${encodeURIComponent(level)}`;
    const res = await fetch(url);
    const data = await res.json();

    state.catalogTotalPages = data.total_pages || 1;
    document.getElementById("catalogCurrentPage").innerText = data.page;
    document.getElementById("catalogPrevBtn").disabled = (data.page <= 1);
    document.getElementById("catalogNextBtn").disabled = (data.page >= data.total_pages);

    const startIdx = (data.page - 1) * data.per_page + 1;
    const endIdx = Math.min(data.page * data.per_page, data.total);
    document.getElementById("catalogPaginationInfo").innerText = `Showing ${startIdx} to ${endIdx} of ${data.total} schemes`;

    const tbody = document.getElementById("catalogTableBody");
    const list = data.schemes || [];

    if (list.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="px-4 py-6 text-center text-slate-400">No schemes found matching catalog filters</td></tr>`;
      return;
    }

    tbody.innerHTML = list.map(s => `
      <tr class="hover:bg-slate-50 transition">
        <td class="px-4 py-3 font-mono text-[11px] font-bold text-sky-700">${s.id}</td>
        <td class="px-4 py-3 font-semibold text-slate-900 cursor-pointer hover:text-sky-600" onclick="openSchemeModal('${s.id}')">${s.name}</td>
        <td class="px-4 py-3 text-slate-600">${s.category}</td>
        <td class="px-4 py-3"><span class="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700">${s.level}</span></td>
        <td class="px-4 py-3 text-slate-500 text-[11px] max-w-xs truncate">${s.department}</td>
        <td class="px-4 py-3 text-right">
          <button onclick="openSchemeModal('${s.id}')" class="px-2.5 py-1 text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium rounded transition">
            View
          </button>
        </td>
      </tr>
    `).join("");

  } catch (err) {
    console.error("Catalog load error:", err);
  }
}

// ----------------- MODAL DOSSIER -----------------
async function openSchemeModal(docId) {
  try {
    const res = await fetch(`/api/schemes/${encodeURIComponent(docId)}`);
    const sch = await res.json();

    document.getElementById("modalIdBadge").innerText = sch.id;
    document.getElementById("modalCategoryBadge").innerText = sch.category;
    document.getElementById("modalLevelBadge").innerText = sch.level;
    document.getElementById("modalSchemeName").innerText = sch.name;
    document.getElementById("modalDepartment").innerText = sch.department || "Government Department";

    document.getElementById("modalTarget").innerText = sch.target_beneficiaries || "N/A";
    document.getElementById("modalDescription").innerText = sch.description || "N/A";
    document.getElementById("modalEligibility").innerText = sch.eligibility || "N/A";
    document.getElementById("modalBenefits").innerText = sch.benefits || "N/A";
    document.getElementById("modalDocuments").innerText = sch.required_documents || "N/A";
    document.getElementById("modalProcedure").innerText = sch.application_procedure || "N/A";

    const officialLink = document.getElementById("modalOfficialLink");
    if (sch.official_portal) {
      officialLink.href = sch.official_portal;
      officialLink.classList.remove("hidden");
    } else {
      officialLink.classList.add("hidden");
    }

    document.getElementById("schemeDetailModal").classList.remove("hidden");
    document.body.style.overflow = "hidden";

  } catch (err) {
    console.error("Scheme detail modal error:", err);
  }
}

function closeSchemeModal() {
  document.getElementById("schemeDetailModal").classList.add("hidden");
  document.body.style.overflow = "auto";
}
