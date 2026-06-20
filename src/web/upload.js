const uploadBtn = document.getElementById('uploadBtn');

uploadBtn.addEventListener('click', async () => {
  const fileInput = document.getElementById('videoInput');
  const statusText = document.getElementById('status');

  if (fileInput.files.length === 0) return;

  uploadBtn.disabled = true;

  const file = fileInput.files[0];
  const chunkSize = 5 * 1024 * 1024; // 5MiB
  const totalChunks = Math.ceil(file.size / chunkSize);

  // TODO(paoda): encodeURIComponent file name

  statusText.innerText = 'Uploading...';

  try {
    for (let i = 0; i < totalChunks; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, file.size);
      const chunk = file.slice(start, end);

      const response = await fetch('/upload', {
        method: 'POST',
        body: chunk,
        headers: {
          'X-Filename': file.name,
          'X-Chunk-Index': `${i}`,
          'X-Total-Chunks': `${totalChunks}`
        }
      });

      if (!response.ok) throw new Error(`Chunk ${i} failed to upload`);

      const percent = Math.round(((i + 1) / totalChunks) * 100);
      statusText.innerText = `Uploading... ${percent}%`;
    }

    statusText.innerText = 'Upload complete!';
    fileInput.value = '';
  } catch (error) {
    statusText.innerText = 'Upload failed or interrupted';
    console.error(error);
  } finally {
    uploadBtn.disabled = false;
  }

});
