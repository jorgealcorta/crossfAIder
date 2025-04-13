import React from 'react';
import './TransitionZone.css';

function TransitionZone({ length, setLength }) {
  return (
    <div className="transition-zone">
      <div className="transition-marker"></div>
      <div className="transition-settings">
        <span className="transition-label">Transición</span>
        <div className="transition-control">
          <input 
            type="range" 
            min="1" 
            max="30" 
            value={length} 
            onChange={(e) => setLength(parseInt(e.target.value))} 
            className="transition-slider"
          />
          <span className="transition-value">{length}s</span>
        </div>
      </div>
    </div>
  );
}

export default TransitionZone;