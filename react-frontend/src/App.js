import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);  // Store predictions array

  const handleFileChange = event => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert('Please upload an image of a bird.');
      return;
    }

    const formData = new FormData();
    formData.append('birdImage', file);

    try {
      const response = await fetch('http://localhost:5000/api/identify', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log("Received response: ", data);
      setPredictions(data);  // Save the full response
    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Failed to upload image.');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <p>Bird Classification</p>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button onClick={handleSubmit}>Upload</button>
        <div>
          {predictions.map((pred, index) => (
            <div key={index}>
              <strong>{pred.species}</strong>: {pred.probability.toFixed(2)}
            </div>
          ))}
        </div>
      </header>
    </div>
  );
}

export default App;
