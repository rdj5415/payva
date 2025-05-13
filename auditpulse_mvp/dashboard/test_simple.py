#!/usr/bin/env python3
"""Simple tests for the AuditPulse dashboard.

These tests can run independently without requiring the full test infrastructure.
"""
import sys
import os
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import dashboard components
from auditpulse_mvp.dashboard.dashboard import (
    AnomalyData,
    export_to_csv,
    export_to_pdf,
    generate_risk_chart,
)


def create_mock_anomalies():
    """Create mock anomaly data for testing."""
    return [
        AnomalyData(
            id=uuid.uuid4(),
            description="Unusual transaction amount",
            risk_score=85.5,
            risk_level="high",
            explanation="This transaction is 4x larger than typical transactions with this vendor.",
            is_resolved=False,
            created_at=datetime.fromisoformat("2023-10-01T14:30:00"),
            transaction_amount=5000.00,
            transaction_date=datetime.fromisoformat("2023-10-01T14:00:00"),
            transaction_description="Payment to XYZ Consulting",
            feedback=None,
        ),
        AnomalyData(
            id=uuid.uuid4(),
            description="Weekend transaction",
            risk_score=65.0,
            risk_level="medium",
            explanation="This transaction occurred on a weekend, which is unusual for this account.",
            is_resolved=False,
            created_at=datetime.fromisoformat("2023-10-02T10:15:00"),
            transaction_amount=1200.00,
            transaction_date=datetime.fromisoformat("2023-10-02T10:00:00"),
            transaction_description="Office supplies purchase",
            feedback=None,
        ),
    ]


def test_export_to_csv():
    """Test the export_to_csv function."""
    mock_anomalies = create_mock_anomalies()
    csv_data = export_to_csv(mock_anomalies)

    # Basic validation
    assert isinstance(csv_data, str)
    assert "ID,Date,Description,Risk Score,Risk Level,Transaction Amount" in csv_data

    # Check content
    assert "high" in csv_data
    assert "medium" in csv_data
    assert "5000.0" in csv_data
    assert "1200.0" in csv_data

    print("✅ CSV export test passed")


def test_export_to_pdf():
    """Test the export_to_pdf function."""
    mock_anomalies = create_mock_anomalies()
    pdf_bytes = export_to_pdf(mock_anomalies)

    # Basic validation
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF-")  # PDF signature

    print("✅ PDF export test passed")


def test_risk_chart():
    """Test risk chart generation."""
    mock_anomalies = create_mock_anomalies()
    chart = generate_risk_chart(mock_anomalies)

    # Validate chart
    assert chart is not None

    # Get risk levels from data
    risk_levels = {}
    for anomaly in mock_anomalies:
        risk_levels[anomaly.risk_level] = risk_levels.get(anomaly.risk_level, 0) + 1

    # Check chart data
    chart_data = chart.data[0]
    chart_x = list(chart_data["x"])
    chart_y = list(chart_data["y"])

    # Check risk levels
    for risk_level, count in risk_levels.items():
        assert risk_level in chart_x
        index = chart_x.index(risk_level)
        assert chart_y[index] == count

    print("✅ Risk chart test passed")


def main():
    """Run all tests."""
    print("🧪 Running dashboard component tests...")
    test_export_to_csv()
    test_export_to_pdf()
    test_risk_chart()
    print("🎉 All tests passed!")


if __name__ == "__main__":
    main()
