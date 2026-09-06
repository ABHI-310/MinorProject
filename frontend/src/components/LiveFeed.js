import React, { useState, useRef } from 'react';
import axios from 'axios';

function LiveFeed({ equipmentId, onDetectionComplete }) {
  const [detections, setDetections] = useState([]);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setError(null);
    setImagePreview(URL.createObjectURL(file));
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('equipment_id', equipmentId);
    formData.append('model_type', 'adapted');

    try {
      const response = await axios.post('http://localhost:8000/detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setDetections(response.data.detections || []);
      setInferenceTime(response.data.inference_time_ms);
      if (onDetectionComplete) {
        onDetectionComplete(response.data);
      }
    } catch (err) {
      console.error('Error during defect detection inference:', err);
      setError('Failed to reach backend API. Ensure backend is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="live-feed-card">
      <div className="feed-actions">
        <button
          className="btn-upload"
          onClick={() => fileInputRef.current.click()}
          disabled={loading}
        >
          {loading ? 'Analyzing Board...' : '📷 Inspect PCB Image'}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          hidden
        />
        {inferenceTime !== null && (
          <span className="latency-tag">⚡ {inferenceTime.toFixed(1)} ms</span>
        )}
      </div>

      {error && <div className="feed-error">{error}</div>}

      <div className="image-stage">
        {imagePreview ? (
          <div className="preview-container">
            <img src={imagePreview} alt="PCB Inspection" className="inspection-img" />
          </div>
        ) : (
          <div className="empty-preview">
            <p>Upload a PCB image or connect Raspberry Pi edge stream to detect defects.</p>
          </div>
        )}
      </div>

      <div className="detections-container">
        <h4>Detected Defects ({detections.length})</h4>
        {detections.length === 0 ? (
          <p className="no-defects">No defects detected. Board is compliant.</p>
        ) : (
          <div className="detections-list">
            {detections.map((det, idx) => (
              <div key={idx} className={`detection-pill defect-${det.class_name}`}>
                <span className="class-badge">{det.class_name.replace('_', ' ')}</span>
                <span className="conf-value">{(det.confidence * 100).toFixed(1)}%</span>
                {det.bbox && (
                  <span className="bbox-info">
                    [{det.bbox.map((v) => Math.round(v)).join(', ')}]
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default LiveFeed;

