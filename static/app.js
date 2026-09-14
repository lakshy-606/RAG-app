// Minimal client for the two API calls this UI needs: /ingest and /query.
// No framework/build step on purpose — this page is meant to be readable
// end-to-end, not a production frontend.

let lastDocId = null;

document.getElementById("ingest-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("pdf-file");
  const status = document.getElementById("ingest-status");

  if (!fileInput.files.length) {
    status.textContent = "Pick a PDF file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  status.textContent = "Ingesting... (parsing, chunking, embedding — can take a while for larger PDFs)";
  try {
    const res = await fetch("/ingest", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    lastDocId = data.doc_id;
    status.textContent = `Ingested "${data.source_filename}" — ${data.num_chunks} chunks across ${data.num_pages ?? "?"} pages. doc_id: ${data.doc_id}`;
  } catch (err) {
    status.textContent = `Ingest failed: ${err.message}`;
  }
});

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
      body: JSON.stringify({ query: queryInput.value, doc_id: lastDocId }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    answerEl.textContent = data.answer;
    sourcesEl.innerHTML = data.sources
      .map(
        (s) =>
          `<li><strong>p.${s.page ?? "?"}</strong>${s.heading ? ` — "${s.heading}"` : ""}: ${s.snippet.slice(0, 160)}...</li>`
      )
      .join("");
  } catch (err) {
    answerEl.textContent = `Query failed: ${err.message}`;
  }
});
