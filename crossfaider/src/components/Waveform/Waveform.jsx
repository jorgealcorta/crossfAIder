import React, { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import './Waveform.css';

function Waveform({ audioUrl }) {
  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  
  useEffect(() => {
    if (!audioUrl || !waveformRef.current) return;
    
    
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#4169E1',
      progressColor: '#7B68EE', 
      cursorColor: 'transparent',
      height: 80,
      
      barWidth: 0,
      barGap: 0,
      barRadius: 0,
      
      normalize: true,
      
      minPxPerSec: 50,
      fillParent: true
    });
    
    
    wavesurferRef.current = wavesurfer;
    
    
    wavesurfer.load(audioUrl);
    
    
    return () => {
      if (wavesurferRef.current) {
        
        try {
          if (wavesurferRef.current.cancelAjax) {
            wavesurferRef.current.cancelAjax();
          }
          wavesurferRef.current.destroy();
        } catch (e) {
          console.warn("Error al destruir wavesurfer:", e);
        }
        wavesurferRef.current = null;
      }
    };
  }, [audioUrl]);
  
  return (
    <div className="waveform-container">
      <div ref={waveformRef} className="waveform"></div>
    </div>
  );
}

export default Waveform;