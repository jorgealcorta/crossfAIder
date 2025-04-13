import React, { useState } from 'react';
import Header from './components/Header/Header';
import AudioTrack from './components/AudioTrack/AudioTrack';
import TransitionZone from './components/TransitionZone/TransitionZone';
import Controls from './components/Controls/Controls';
import { generateTransition } from './services/api';
import './App.css';

function App() {
  const [trackOne, setTrackOne] = useState(null);
  const [trackTwo, setTrackTwo] = useState(null);
  const [transitionLength, setTransitionLength] = useState(5); 
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerateTransition = async () => {
    if (!trackOne || !trackTwo) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      
      const resultUrl = await generateTransition(
        trackOne.file, 
        trackTwo.file, 
        transitionLength
      );
      
      setResult({
        url: resultUrl,
        name: `${trackOne.name.split('.')[0]}_to_${trackTwo.name.split('.')[0]}.mp3`
      });
    } catch (err) {
      console.error('Error generando transición:', err);
      setError('Hubo un error al generar la transición. Por favor intenta de nuevo.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <Header />
      
      <main className="main-content">
        <div className="timeline-container">
          <AudioTrack 
            position="left"
            track={trackOne} 
            setTrack={setTrackOne} 
            label="Primera pista" 
          />
          
          <TransitionZone 
            length={transitionLength}
            setLength={setTransitionLength}
          />
          
          <AudioTrack 
            position="right"
            track={trackTwo} 
            setTrack={setTrackTwo} 
            label="Segunda pista" 
          />
        </div>
        
        {error && <div className="error-message">{error}</div>}
        
        <Controls 
          trackOne={trackOne}
          trackTwo={trackTwo}
          isProcessing={isProcessing}
          onGenerate={handleGenerateTransition}
          result={result}
        />
        
        {result && (
          <div className="result-player">
            <h3>Resultado</h3>
            <audio controls src={result.url}></audio>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;