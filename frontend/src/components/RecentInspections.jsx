import React, { useState, useMemo } from 'react';
import {
  Search,
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Eye,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { recentInspections } from '../data/mockDetections';
import './RecentInspections.css';

export default function RecentInspections({ onSelectInspection }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 5;

  const filteredInspections = useMemo(() => {
    return recentInspections.filter((item) => {
      const matchesSearch =
        item.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.defectType.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.details.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesStatus =
        statusFilter === 'all' ||
        (statusFilter === 'defect' && item.statusType === 'defect') ||
        (statusFilter === 'review' && item.statusType === 'review') ||
        (statusFilter === 'passed' && item.statusType === 'passed');

      return matchesSearch && matchesStatus;
    });
  }, [searchQuery, statusFilter]);

  const totalPages = Math.ceil(filteredInspections.length / pageSize) || 1;
  const paginatedInspections = filteredInspections.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  const toggleStatusFilter = () => {
    const sequence = ['all', 'defect', 'review', 'passed'];
    const nextIdx = (sequence.indexOf(statusFilter) + 1) % sequence.length;
    setStatusFilter(sequence[nextIdx]);
    setCurrentPage(1);
  };

  const getStatusLabel = () => {
    if (statusFilter === 'all') return 'All Status';
    if (statusFilter === 'defect') return 'Defect Found';
    if (statusFilter === 'review') return 'Review Req.';
    if (statusFilter === 'passed') return 'Passed';
    return 'Filter';
  };

  return (
    <div className="ui-card inspections-card">
      {/* Header */}
      <div className="inspections-card__header">
        <div className="inspections-card__title-left">
          <span className="live-stream-dot"></span>
          <h2 className="inspections-card__title">Recent Inspections</h2>
          <span className="inspections-card__stream-badge">Live Stream (30s interval)</span>
        </div>
      </div>

      {/* Search & Filter Bar */}
      <div className="inspections-card__toolbar">
        <div className="inspections-search">
          <Search size={14} className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Filter by ID or defect..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
          />
        </div>

        <button
          className={`inspections-filter-btn ${statusFilter !== 'all' ? 'inspections-filter-btn--active' : ''}`}
          onClick={toggleStatusFilter}
          title="Click to cycle status filters"
        >
          <SlidersHorizontal size={13} />
          <span>{getStatusLabel()}</span>
        </button>
      </div>

      {/* Table */}
      <div className="inspections-table-wrap">
        <table className="inspections-table">
          <thead>
            <tr>
              <th style={{ width: '80px' }}>THUMBNAIL</th>
              <th>INSPECTION ID</th>
              <th>DEFECT TYPE</th>
              <th>CONFIDENCE</th>
              <th>TIMESTAMP</th>
              <th>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {paginatedInspections.map((item) => (
              <tr
                key={item.id}
                className="inspection-row"
                onClick={() => onSelectInspection && onSelectInspection(item)}
                title="Click to view detailed PCB inspection"
              >
                {/* Thumbnail */}
                <td>
                  <div className="inspection-thumb-box">
                    <img
                      src={item.thumbnail}
                      alt={item.id}
                      className="inspection-thumb-img"
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                    <div className="inspection-thumb-fallback">
                      <span className="thumb-grid"></span>
                    </div>
                  </div>
                </td>

                {/* ID */}
                <td>
                  <span className="inspection-id">{item.id}</span>
                </td>

                {/* Defect Type Pill */}
                <td>
                  <span
                    className="defect-type-pill"
                    style={{
                      backgroundColor: item.badgeBg,
                      color: item.color === '#ef4444' ? '#b91c1c' : item.color === '#10b981' ? '#15803d' : '#9a3412',
                    }}
                  >
                    <span className="defect-pill-dot" style={{ backgroundColor: item.color }}></span>
                    <span>{item.defectType}</span>
                  </span>
                </td>

                {/* Confidence Bar */}
                <td>
                  <div className="confidence-cell">
                    <div className="confidence-track">
                      <div
                        className="confidence-fill"
                        style={{
                          width: `${item.confidence}%`,
                          backgroundColor: item.statusType === 'passed' ? '#15803d' : '#78350f'
                        }}
                      ></div>
                    </div>
                    <span className="confidence-text">{item.confidence}%</span>
                  </div>
                </td>

                {/* Timestamp */}
                <td>
                  <div className="timestamp-cell">
                    <span className="timestamp-time">{item.timestamp}</span>
                    <span className="timestamp-ago">{item.timeAgo}</span>
                  </div>
                </td>

                {/* Status Badge */}
                <td>
                  <span className={`status-badge status-badge--${item.statusType}`}>
                    {item.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="inspections-card__footer">
        <span className="inspections-count-text">
          Showing {paginatedInspections.length} of 1,284 inspections
        </span>

        <div className="inspections-pagination">
          <button
            className="pagination-btn"
            disabled={currentPage <= 1}
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
          >
            Previous
          </button>
          <span className="pagination-page">
            Page {currentPage} of 257
          </span>
          <button
            className="pagination-btn"
            disabled={currentPage >= totalPages && currentPage >= 257}
            onClick={() => setCurrentPage((p) => p + 1)}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
