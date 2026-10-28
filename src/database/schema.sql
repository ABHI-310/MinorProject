-- PostgreSQL schema generated from the application ORM. Existing tables are never dropped.


CREATE TABLE IF NOT EXISTS detections (
	id SERIAL NOT NULL, 
	equipment_id VARCHAR(50) NOT NULL, 
	image_path VARCHAR(255), 
	defect_class VARCHAR(50) NOT NULL, 
	bbox_coords TEXT NOT NULL, 
	confidence FLOAT NOT NULL, 
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
)

;

CREATE INDEX IF NOT EXISTS ix_detections_equipment_id ON detections (equipment_id);

CREATE INDEX IF NOT EXISTS ix_detections_timestamp ON detections (timestamp);


CREATE TABLE IF NOT EXISTS inspections (
	id VARCHAR(36) NOT NULL, 
	equipment_id VARCHAR(50) NOT NULL, 
	request_hash VARCHAR(64) NOT NULL, 
	model_type VARCHAR(20) NOT NULL, 
	image_path VARCHAR(255), 
	boards_inspected INTEGER, 
	defect_count INTEGER NOT NULL, 
	detections_json TEXT NOT NULL, 
	details_json TEXT NOT NULL, 
	rul_json TEXT NOT NULL, 
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
)

;

CREATE INDEX IF NOT EXISTS ix_inspections_equipment_id ON inspections (equipment_id);

CREATE INDEX IF NOT EXISTS ix_inspections_timestamp ON inspections (timestamp);


CREATE TABLE IF NOT EXISTS model_metrics (
	id SERIAL NOT NULL, 
	model_name VARCHAR(100) NOT NULL, 
	model_type VARCHAR(50) NOT NULL, 
	accuracy FLOAT, 
	precision FLOAT, 
	recall FLOAT, 
	map_score FLOAT, 
	f1_score FLOAT, 
	inference_time_ms FLOAT, 
	evaluated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
)

;


CREATE TABLE IF NOT EXISTS rul_estimates (
	id SERIAL NOT NULL, 
	equipment_id VARCHAR(50) NOT NULL, 
	health_score FLOAT NOT NULL, 
	rul_cycles INTEGER NOT NULL, 
	defect_count INTEGER NOT NULL, 
	synthetic BOOLEAN NOT NULL, 
	predicted_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
)

;

CREATE INDEX IF NOT EXISTS ix_rul_estimates_predicted_at ON rul_estimates (predicted_at);

CREATE INDEX IF NOT EXISTS ix_rul_estimates_equipment_id ON rul_estimates (equipment_id);


CREATE TABLE IF NOT EXISTS verified_models (
	checkpoint_sha256 VARCHAR(64) NOT NULL, 
	model_type VARCHAR(20) NOT NULL, 
	report_json TEXT NOT NULL, 
	registered_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (checkpoint_sha256)
)

;
