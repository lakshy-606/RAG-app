// Client for /config, /reindex, and /query. No framework/build step on
// purpose — this page is meant to be readable end-to-end, not a
// production frontend.
//
// Ingestion (uploading a new PDF) isn't exposed here — Docling's
// cold-start time on Lambda is too slow for a synchronous POST /ingest
// request. Changing the *embedding model*, though, doesn't need Docling
// at all — it re-embeds the already-extracted chunks (see
// app/retriever.reindex_with_model), so that's fast enough to expose live.

const embeddingSelect = document.getElementById("embedding-model");
const chatSelect = document.getElementById("chat-model");
const topKInput = document.getElementById("top-k");
const topKValue = document.getElementById("top-k-value");
const temperatureInput = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");
const useRerankingInput = document.getElementById("use-reranking");
const reindexStatus = document.getElementById("reindex-status");

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});
temperatureInput.addEventListener("input", () => {
  temperatureValue.textContent = parseFloat(temperatureInput.value).toFixed(1);
});

async function loadConfig() {
  const res = await fetch("/config");
  const cfg = await res.json();

  for (const model of cfg.embedding_models) {
    const opt = new Option(model, model, model === cfg.default_embedding_model, model === cfg.default_embedding_model);
    embeddingSelect.add(opt);
  }
  for (const model of cfg.chat_models) {
    const opt = new Option(model, model, model === cfg.default_chat_model, model === cfg.default_chat_model);
    chatSelect.add(opt);
  }
}

// Changing the embedding model is the one setting that isn't purely
// per-query — it determines which Pinecone index gets searched, and that
// index needs to actually have data in it first. Triggers immediately on
// selection (a dropdown's natural "confirm" gesture), not on a separate
// button.
embeddingSelect.addEventListener("change", async () => {
  const model = embeddingSelect.value;
  reindexStatus.textContent = `Switching to ${model}... (re-embedding if this model hasn't been used before)`;
  try {
    const res = await fetch("/reindex", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embedding_model: model }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    reindexStatus.textContent =
      data.status === "reindexed"
        ? `Ready — embedded ${data.num_chunks} chunks with ${model}.`
        : `Ready — ${model} was already indexed.`;
  } catch (err) {
    reindexStatus.textContent = `Failed to switch models: ${err.message}`;
  }
});

// Snippets longer than this collapse behind a native <details> element —
// the <summary> itself shows a truncated preview, so there's always
// something to read before expanding. No custom JS needed for the
// expand/collapse interaction itself: <details> is keyboard-accessible
// and works with zero script by design.
const SNIPPET_COLLAPSE_THRESHOLD = 200;

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderSource(s) {
  const label = `<strong>p.${s.page ?? "?"}</strong>${s.heading ? ` — "${escapeHtml(s.heading)}"` : ""}`;
  const snippet = escapeHtml(s.snippet);

  if (snippet.length <= SNIPPET_COLLAPSE_THRESHOLD) {
    return `<li>${label}: ${snippet}</li>`;
  }

  const preview = escapeHtml(s.snippet.slice(0, SNIPPET_COLLAPSE_THRESHOLD)) + "…";
  return `<li>${label}:
    <details>
      <summary>${preview}</summary>
      <p class="snippet-full">${snippet}</p>
    </details>
  </li>`;
}

document.getElementById("query-btn").addEventListener("click", async () => {
  const queryInput = document.getElementById("query-input");
  const answerEl = document.getElementById("answer");
  const sourcesEl = document.getElementById("sources");

  if (!queryInput.value.trim()) return;

  answerEl.textContent = "Thinking...";
  sourcesEl.innerHTML = "";

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryInput.value,
        embedding_model: embeddingSelect.value || null,
        chat_model: chatSelect.value,
        top_k: parseInt(topKInput.value, 10),
        temperature: parseFloat(temperatureInput.value),
        use_reranking: useRerankingInput.checked,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    answerEl.textContent = data.answer;
    sourcesEl.innerHTML = data.sources.map(renderSource).join("");
  } catch (err) {
    answerEl.textContent = `Query failed: ${err.message}`;
  }
});

document.getElementById("query-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("query-btn").click();
});

loadConfig();
