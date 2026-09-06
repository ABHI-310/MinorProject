-- PostgreSQL Database Schema for PCB Smart Maintenance System

CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) NOT NULL,
    image_path VARCHAR(255),
    defect_class VARCHAR(50) NOT NULL,
    bbox_coords TEXT,  -- JSON formatted: [x1, y1, x2, y2]
    confidence FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rul_estimates (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(50) NOT NULL,
    health_score FLOAT NOT NULL,
    rul_cycles INT NOT NULL,
    defect_count INT NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'source' or 'adapted'
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    map_score FLOAT,
    f1_score FLOAT,
    inference_time_ms FLOAT,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization indexes for high-throughput edge event logging
CREATE INDEX IF NOT EXISTS idx_equipment_id ON detections(equipment_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_rul_equipment ON rul_estimates(equipment_id);
CREATE INDEX IF NOT EXISTS idx_rul_predicted_at ON rul_estimates(predicted_at);

