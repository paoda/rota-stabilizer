document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('videoInput');
    const statusText = document.getElementById('status');

    if (fileInput.files.length === 0) return;

    const file = fileInput.files[0];
    statusText.innerText = "Uploading...";
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: file, 
            headers: {
              'X-Filename': file.name,
            }
        });

        if (response.ok) {
            statusText.innerText = 'Upload complete';
            fileInput.value = ''; // Clear the input
        } else {
            statusText.innerText = 'Upload failed';
        }
    } catch (error) {
        statusText.innerText = 'Network error';
        console.error(error);
    }
});
