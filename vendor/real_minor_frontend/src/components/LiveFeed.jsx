import { useState, useEffect, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Pause, Play, Radio, ImageOff } from 'lucide-react';
import { mockDetections } from '../data/mockDetections';
import './LiveFeed.css';

export default function LiveFeed() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [imageLoaded, setImageLoaded] = useState(true);
  const [imageError, setImageError] = useState(false);

  const detection = mockDetections[currentIndex];

  const goNext = useCallback(() => {
    setCurrentIndex((prev) => (prev + 1) % mockDetections.length);
    setImageLoaded(false);
    setImageError(false);
  }, []);

  const goPrev = useCallback(() => {
    setCurrentIndex((prev) => (prev - 1 + mockDetections.length) % mockDetections.length);
    setImageLoaded(false);
    setImageError(false);
  }, []);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(goNext, 5000);
    return () => clearInterval(timer);
  }, [isPlaying, goNext]);

  const severityClass = detection.severity === 'critical' ? 'badge--critical'
    : detection.severity === 'warning' ? 'badge--warning' : 'badge--info';

  const statusClass = detection.status === 'FAIL' ? 'badge--fail' : 'badge--review';

  return (
    <div className="glass-card live-feed">
      <div className="glass-card__header">
        <div className="glass-card__title">
          <Radio size={16} />
          Live Detection Feed
        </div>
        <span className={`badge ${severityClass}`}>{detection.severity}</span>
      </div>

      <div className="live-feed__viewer">
        {imageError ? (
          <div className="live-feed__no-image">
            <ImageOff size={48} />
            <span style={{ fontSize: '0.8rem' }}>Image not available</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              Copy prediction samples to public/prediction_samples/
            </span>
          </div>
        ) : (
          <img
            className={`live-feed__image ${imageLoaded ? '' : 'live-feed__image--loading'}`}
            src={detection.imagePath}
            alt={`PCB Detection - ${detection.defectLabel}`}
            onLoad={() => setImageLoaded(true)}
            onError={() => setImageError(true)}
          />
        )}

        <div className="live-feed__overlay">
          <span className="live-feed__counter">
            {currentIndex + 1} / {mockDetections.length}
          </span>
          <span className={`badge ${statusClass} live-feed__status-badge`}>
            {detection.status}
          </span>
        </div>

        <div className="live-feed__controls">
          <button className="live-feed__btn" onClick={goPrev} aria-label="Previous">
            <ChevronLeft size={16} />
          </button>
          <button
            className={`live-feed__btn ${isPlaying ? 'live-feed__btn--active' : ''}`}
            onClick={() => setIsPlaying(!isPlaying)}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <button className="live-feed__btn" onClick={goNext} aria-label="Next">
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      <div className="live-feed__details">
        <div className="live-feed__detail-grid">
          <div className="live-feed__detail-item">
            <span className="live-feed__detail-label">Defect Type</span>
            <span className="live-feed__detail-value">{detection.defectLabel}</span>
          </div>
          <div className="live-feed__detail-item">
            <span className="live-feed__detail-label">Board ID</span>
            <span className="live-feed__detail-value live-feed__detail-value--mono">{detection.boardId}</span>
          </div>
          <div className="live-feed__detail-item">
            <span className="live-feed__detail-label">Confidence</span>
            <span className="live-feed__detail-value">{(detection.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="live-feed__detail-item">
            <span className="live-feed__detail-label">Timestamp</span>
            <span className="live-feed__detail-value live-feed__detail-value--mono">
              {new Date(detection.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
