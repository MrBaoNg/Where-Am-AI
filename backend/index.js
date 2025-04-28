const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });
app.use(cors());

app.post('/upload', upload.single('image'), (req, res) => {
  const imagePath = path.resolve(req.file.path);

  exec(`python3 model.py "${imagePath}"`, (error, stdout, stderr) => {
    if (error) {
      console.error('Model error:', stderr);
      return res.status(500).json({ message: 'Error running model.' });
    }

    res.json({ message: stdout.trim() });
  });
});

app.listen(5000, () => {
  console.log('Backend running on port 5000');
});
