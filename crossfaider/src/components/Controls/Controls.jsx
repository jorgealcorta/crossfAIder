import React from 'react';
import './Controls.css';

function Controls({ trackOne, trackTwo, isProcessing, onGenerate, result }) {
  const canGenerate = trackOne && trackTwo && !isProcessing;
  
  const handleDownload = () => {
    if (result && result.url) {
      const a = document.createElement('a');
      a.href = result.url;
      a.download = 'crossfaded_mix.mp3';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };
  
  return (
    <div className="controls-container">
      <button 
        className="preview-button"
        disabled={!(trackOne || trackTwo)}
      >
        Vista previa
      </button>
      
      <button 
        className="generate-button"
        disabled={!canGenerate}
        onClick={onGenerate}
      >
        {isProcessing ? 'Generando...' : 'Generar transición'}
      </button>
      
      {result && (
        <button 
          className="download-button"
          onClick={handleDownload}
        >
          Descargar resultado
        </button>
      )}
    </div>
  );
}

export default Controls;