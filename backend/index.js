const express = require('express');
const cors    = require('cors');
const multer  = require('multer');
const { spawn } = require('child_process');
const path    = require('path');
const fs      = require('fs');

const app    = express();
const upload = multer({ dest: 'uploads/' });
app.use(cors());

function clearUploadsSync() {
  const dir = 'uploads/';
  if (!fs.existsSync(dir)) return;
  for (const file of fs.readdirSync(dir)) {
    try { fs.unlinkSync(path.join(dir, file)); }
    catch (err) { console.warn(`Could not delete ${file}:`, err); }
  }
}

app.post('/upload', upload.single('image'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  const imagePath = path.resolve(req.file.path);
  const py = spawn('python', ['-W','ignore','model.py','--image', imagePath], {
    cwd: __dirname,
    stdio: ['ignore','pipe','pipe']
  });
  let stdout = '', stderr = '';
  py.stdout.on('data', d => stdout += d.toString());
  py.stderr.on('data', d => stderr += d.toString());
  py.on('close', code => {
    clearUploadsSync();
    if (code !== 0) {
      console.error('Model error:', stderr);
      return res.status(500).json({ error: 'Model failed', details: stderr });
    }
  
    // Split into lines and take the **last non-empty** one
    const lines = stdout.trim().split(/\r?\n/).filter(l => l.trim() !== '');
    const last = lines[lines.length - 1];
    const parts = last.trim().split(/\s+/);
  
    if (parts.length !== 2 || parts.some(p => isNaN(p))) {
      console.error('Unexpected model output:', stdout);
      return res.status(500).json({ error: 'Unexpected model output', raw: stdout });
    }
  
    const [lat, lon] = parts.map(Number);
    res.json({
      message: `${lat} ${lon}`
  });
});

app.listen(5000, () => {
  console.log('Backend running on port 5000');
});


app.listen(5000, () => console.log('Backend running on port 5000'));
