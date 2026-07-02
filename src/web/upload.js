import NoSleep from '/lib/nosleep.min.js';

const uploadBtn = document.getElementById('uploadBtn');
const MAX_RETRIES = 4;
const BASE_DELAY_MS = 500;

const noSleep = new NoSleep();

/** 
 * @param {number} ms  
 * @returns {Promise<number>}
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/** 
 * @callback retryCallback
 * @param {number} attempt - current attempt
 * @param {number} max - max attempt
 * @param {number} delay - ms delay before next retry
 * @returns {void}
 */

/**
 * @param {ArrayBuffer} chunk
 * @param {File} file
 * @param {number} index
 * @param {number} offset - byte offset where this chunk starts within the full file
 * @param {retryCallback} onRetry 
 * @returns {Promise<Response>}
 *
 */
async function uploadChunkWithRetry(chunk, file, index, offset, onRetry) {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: chunk,
        headers: {
          'X-Filename': file.name,
          'X-Chunk-Index': `${index}`,
          'X-Chunk-Offset': `${offset}`,
          'X-File-Size': `${file.size}`
        }
      });

      const body = await response.text();
      if (response.ok) return response;

      const error = new Error(`Chunk ${index} failed: HTTP ${response.status} (${body})`);
      error.retryable = response.status >= 500;

      throw error;
    } catch (error) {
      if (attempt === MAX_RETRIES || error.retryable === false) throw error;

      const delay_ms = BASE_DELAY_MS * Math.pow(2, attempt) + Math.random() * 200;
      onRetry(attempt + 1, MAX_RETRIES, delay_ms);

      await sleep(delay_ms);
    }
  }
}

uploadBtn.addEventListener('click', async () => {
  /** @type {HTMLInputElement | null} */
  const fileInput = document.getElementById('videoInput');
  if (!fileInput) throw new Error('missing <input id="videoInput" />')

  const statusText = document.getElementById('status');
  if (!statusText) throw new Error('missing <p id="status" />');

  if (fileInput.files.length === 0) return;

  uploadBtn.disabled = true;

  const file = fileInput.files[0];
  const chunkSize = 5 * 1024 * 1024; // 5MiB
  const totalChunks = Math.ceil(file.size / chunkSize);

  // TODO(paoda): encodeURIComponent file name
  statusText.innerText = 'Uploading...';
  noSleep.enable();

  try {
    for (let i = 0; i < totalChunks; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, file.size);

      // WebKit wedges on file-backed Blob bodies; materialize first
      const chunk = await file.slice(start, end).arrayBuffer();
      await uploadChunkWithRetry(chunk, file, i, start, (attempt, max, delay) => {

        statusText.innerText = `Chunk ${i} failed, retrying (${attempt}/${max}) in ${Math.round(delay / 1000)}s...`;
      });

      const percent = Math.round(((i + 1) / totalChunks) * 100);
      statusText.innerText = `Uploading... ${percent}%`;
    }

    statusText.innerText = 'Upload complete!';
    fileInput.value = '';
  } catch (error) {
    statusText.innerText = `Upload failed: ${error.message}`;
    console.error(error);
  } finally {
    noSleep.disable();
    uploadBtn.disabled = false;
  }
});

const logEl = document.getElementById('log');
const t0 = performance.now();

function log(msg) {
  logEl.textContent += `[${((performance.now() - t0) / 1000).toFixed(3)}] ${msg}\n`;
}

document.addEventListener('visibilitychange', () => log(`visibility: ${document.visibilityState}`));
window.addEventListener('pagehide', () => log('pagehide'));
