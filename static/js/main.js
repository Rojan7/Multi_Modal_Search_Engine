const textForm = document.getElementById("text-search-form");
const imageForm = document.getElementById("image-search-form");
const imageInput = document.getElementById("image-input");
const fileName = document.getElementById("file-name");
const trainButton = document.getElementById("train-button");
const statusText = document.getElementById("status-text");
const resultsContainer = document.getElementById("results");
const emptyState = document.getElementById("empty-state");
const errorBanner = document.getElementById("error-banner");
const resultCount = document.getElementById("result-count");

function setStatus(message) {
    statusText.textContent = message;
}

function setError(message) {
    if (!message) {
        errorBanner.classList.add("hidden");
        errorBanner.textContent = "";
        return;
    }

    errorBanner.textContent = message;
    errorBanner.classList.remove("hidden");
}

function clearResults() {
    resultsContainer.innerHTML = "";
    resultCount.textContent = "0 results";
    emptyState.classList.remove("hidden");
}

function renderResults(results) {
    resultsContainer.innerHTML = "";

    if (!results || results.length === 0) {
        resultCount.textContent = "0 results";
        emptyState.textContent = "No matches were returned for this request.";
        emptyState.classList.remove("hidden");
        return;
    }

    emptyState.classList.add("hidden");
    resultCount.textContent = `${results.length} result${results.length === 1 ? "" : "s"}`;

    results.forEach((result, index) => {
        const card = document.createElement("article");
        card.className = "result-card";

        const score = typeof result.score === "number" ? result.score.toFixed(4) : "N/A";
        const url = result.url || "No URL available";

        card.innerHTML = `
            <div class="flex items-start justify-between gap-3">
                <span class="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-slate-600">
                    Match ${index + 1}
                </span>
                <span class="rounded-full bg-emerald-50 px-3 py-1 text-sm font-medium text-emerald-700">
                    Score ${score}
                </span>
            </div>
            <p class="mt-4 break-all text-sm leading-6 text-slate-600">${url}</p>
            <a class="mt-4 inline-flex text-sm font-medium text-teal-700 underline underline-offset-4" href="${url}" target="_blank" rel="noreferrer">
                Open source page
            </a>
        `;

        resultsContainer.appendChild(card);
    });
}

async function postForm(url, formData) {
    const response = await fetch(url, {
        method: "POST",
        body: formData
    });

    return response.json();
}

textForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setError("");

    const query = document.getElementById("query-input").value.trim();
    if (!query) {
        setError("Enter a text query before searching.");
        return;
    }

    setStatus("Running text search...");

    try {
        const formData = new FormData();
        formData.append("query", query);

        const payload = await postForm("/api/search/text", formData);
        if (payload.error) {
            clearResults();
            setError(payload.error);
            setStatus("Search returned a setup error.");
            return;
        }

        renderResults(payload.results);
        setStatus("Text search completed.");
    } catch (error) {
        clearResults();
        setError("The request failed before a response was returned.");
        setStatus("Search request failed.");
    }
});

imageInput.addEventListener("change", () => {
    const selectedFile = imageInput.files[0];
    fileName.textContent = selectedFile ? selectedFile.name : "No file selected";
});

imageForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setError("");

    const selectedFile = imageInput.files[0];
    if (!selectedFile) {
        setError("Choose an image before searching.");
        return;
    }

    setStatus("Running image search...");

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const payload = await postForm("/api/search/image", formData);
        if (payload.error) {
            clearResults();
            setError(payload.error);
            setStatus("Image search returned a setup error.");
            return;
        }

        renderResults(payload.results);
        setStatus("Image search completed.");
    } catch (error) {
        clearResults();
        setError("The image request failed before a response was returned.");
        setStatus("Image search request failed.");
    }
});

trainButton.addEventListener("click", async () => {
    setError("");
    setStatus("Training pipeline started...");

    try {
        const response = await fetch("/train");
        const message = await response.text();

        if (!response.ok || message.startsWith("Error:")) {
            setError(message);
            setStatus("Training request failed.");
            return;
        }

        setStatus(message);
    } catch (error) {
        setError("The training request could not reach the backend.");
        setStatus("Training request failed.");
    }
});