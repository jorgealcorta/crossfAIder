import React, { useRef, useState } from 'react';
import Waveform from '../Waveform/Waveform';
import './AudioTrack.css';

function AudioTrack({ position, track, setTrack, label }) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.includes('audio')) {
      handleFile(file);
    }
  };
  
  const handleFile = (file) => {
    const audioObj = new Audio();
    const objectUrl = URL.createObjectURL(file);
    audioObj.src = objectUrl;
    
    
    audioObj.onloadedmetadata = () => {
      setTrack({
        file,
        name: file.name,
        duration: audioObj.duration,
        url: objectUrl
      });
      
      
      audioObj.onloadedmetadata = null;
    };
  };
  
  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.includes('audio')) {
      handleFile(file);
    }
  };
  
  const handleRemoveTrack = () => {
    if (track && track.url) {
      URL.revokeObjectURL(track.url);
    }
    setTrack(null);
  };
  
  return (
    <div 
      className={`audio-track ${position} ${isDragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {track ? (
        <div className="track-content">
          <Waveform audioUrl={track.url} />
          <div className="track-info">
            <p className="track-name">{track.name}</p>
            <button className="remove-track" onClick={handleRemoveTrack}>
              ✕
            </button>
          </div>
        </div>
      ) : (
        <div className="upload-zone">
          <p>{label}</p>
          <button 
            className="upload-button"
            onClick={() => fileInputRef.current.click()}
          >
            Seleccionar archivo
          </button>
          <input 
            ref={fileInputRef}
            type="file" 
            accept="audio/*" 
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <p className="drag-hint">o arrastra aquí</p>
        </div>
      )}
    </div>
  );
}

export default AudioTrack;