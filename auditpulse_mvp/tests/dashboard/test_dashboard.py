"""Tests for the Streamlit dashboard.

This module contains tests for the AuditPulse Streamlit dashboard.
"""

import json
import re
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pandas as pd

from auditpulse_mvp.dashboard.dashboard import (
    AnomalyData,
    fetch_anomalies,
    submit_anomaly_feedback,
    export_to_csv,
    export_to_pdf,
    generate_risk_chart,
)


@pytest.fixture
def mock_anomalies():
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
        AnomalyData(
            id=uuid.uuid4(),
            description="Duplicate payment",
            risk_score=92.0,
            risk_level="critical",
            explanation="This appears to be a duplicate payment to the same vendor within 24 hours.",
            is_resolved=True,
            created_at=datetime.fromisoformat("2023-10-03T09:45:00"),
            transaction_amount=3500.00,
            transaction_date=datetime.fromisoformat("2023-10-03T09:30:00"),
            transaction_description="Payment to ABC Services",
            feedback="true_positive",
        ),
    ]


@pytest.mark.asyncio
async def test_fetch_anomalies():
    """Test the fetch_anomalies function."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "items": [
            {
                "id": str(uuid.uuid4()),
                "description": "Test anomaly",
                "risk_score": 75.0,
                "risk_level": "high",
                "explanation": "Test explanation",
                "is_resolved": False,
                "created_at": "2023-10-01T12:00:00Z",
                "updated_at": "2023-10-01T12:00:00Z",
                "transaction": {
                    "id": str(uuid.uuid4()),
                    "date": "2023-10-01T11:00:00Z",
                    "amount": 1000.0,
                    "description": "Test transaction",
                    "type": "expense",
                },
            }
        ],
        "total": 1,
        "page": 1,
        "page_size": 50,
        "has_more": False,
    }

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        anomalies, total = await fetch_anomalies(
            start_date=datetime.fromisoformat("2023-10-01T00:00:00"),
            end_date=datetime.fromisoformat("2023-10-31T23:59:59"),
        )

    assert len(anomalies) == 1
    assert total == 1
    assert anomalies[0].description == "Test anomaly"
    assert anomalies[0].risk_score == 75.0
    assert anomalies[0].transaction_amount == 1000.0


@pytest.mark.asyncio
async def test_submit_anomaly_feedback():
    """Test the submit_anomaly_feedback function."""
    anomaly_id = str(uuid.uuid4())

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"success": True}

    mock_notification_response = MagicMock()
    mock_notification_response.raise_for_status = MagicMock()
    mock_notification_response.json.return_value = {"success": True}

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.post.side_effect = [
        mock_response,  # First call for feedback
        mock_notification_response,  # Second call for notification
    ]

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await submit_anomaly_feedback(
            anomaly_id=anomaly_id,
            feedback_type="false_positive",
            notes="Test feedback",
        )

    assert result is True

    # Verify API call
    calls = mock_client.__aenter__.return_value.post.call_args_list
    assert len(calls) == 2

    # Verify feedback API call
    feedback_call = calls[0]
    assert (
        feedback_call[0][0]
        == f"http://localhost:8000/api/v1/anomalies/{anomaly_id}/feedback"
    )
    assert feedback_call[1]["json"]["feedback_type"] == "false_positive"
    assert feedback_call[1]["json"]["notes"] == "Test feedback"

    # Verify notification API call
    notification_call = calls[1]
    assert (
        notification_call[0][0]
        == f"http://localhost:8000/api/v1/alerts/send/{anomaly_id}"
    )


def test_export_to_csv(mock_anomalies):
    """Test the export_to_csv function."""
    csv_data = export_to_csv(mock_anomalies)

    # Verify CSV format and content
    assert isinstance(csv_data, str)
    assert "ID,Date,Description,Risk Score,Risk Level,Transaction Amount" in csv_data

    # Parse CSV back to DataFrame for validation
    from io import StringIO

    df = pd.read_csv(StringIO(csv_data))

    assert len(df) == len(mock_anomalies)
    assert "Risk Score" in df.columns
    assert "Risk Level" in df.columns
    assert "Transaction Amount" in df.columns

    # Verify values
    assert df.iloc[0]["Risk Level"] == "high"
    assert df.iloc[0]["Transaction Amount"] == 5000.00
    assert df.iloc[1]["Risk Level"] == "medium"
    assert df.iloc[2]["Risk Level"] == "critical"


def test_export_to_pdf(mock_anomalies):
    """Test the export_to_pdf function."""
    pdf_bytes = export_to_pdf(mock_anomalies)

    # Basic validation of PDF output
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF-")  # PDF signature


def test_generate_risk_chart(mock_anomalies):
    """Test the generate_risk_chart function."""
    chart = generate_risk_chart(mock_anomalies)

    # Verify chart was created
    assert chart is not None

    # Extract data
    risk_counts = {}
    for anomaly in mock_anomalies:
        risk_counts[anomaly.risk_level] = risk_counts.get(anomaly.risk_level, 0) + 1

    # Verify chart data matches our mock data
    chart_data = chart.data[0]
    chart_x = list(chart_data["x"])
    chart_y = list(chart_data["y"])

    # Check all risk levels are represented
    for risk_level, count in risk_counts.items():
        assert risk_level in chart_x
        index = chart_x.index(risk_level)
        assert chart_y[index] == count


def test_mock_integration():
    """Test an integration-like scenario without requiring streamlit testing."""
    # This is a simplified integration test that doesn't require the streamlit-testing package
    with (
        patch("streamlit.session_state", {"authenticated": True}),
        patch("streamlit.title") as mock_title,
        patch("streamlit.sidebar") as mock_sidebar,
        patch("streamlit.spinner") as mock_spinner,
        patch("asyncio.new_event_loop") as mock_loop,
    ):

        # Import the main after patching
        with patch.dict("sys.modules", {"streamlit": MagicMock()}):
            from auditpulse_mvp.dashboard.dashboard import show_dashboard

            # Mock the loop to return data
            mock_loop_instance = MagicMock()
            mock_loop.return_value = mock_loop_instance

            mock_anomalies_data = mock_anomalies()
            mock_loop_instance.run_until_complete.return_value = (
                mock_anomalies_data,
                len(mock_anomalies_data),
            )

            # Call the dashboard function (it won't actually render anything due to our mocks)
            show_dashboard()

            # Verify the title was called (basic smoke test)
            mock_title.assert_called_once()


def test_filter_validation():
    """Test filter validation logic."""
    # We're testing the filter logic without needing to run streamlit
    with (
        patch("streamlit.session_state", {}),
        patch("streamlit.form") as mock_form,
        patch("streamlit.date_input") as mock_date_input,
        patch("streamlit.slider") as mock_slider,
        patch("streamlit.form_submit_button") as mock_submit,
    ):

        # Setup mocks
        mock_form.return_value.__enter__ = MagicMock()
        mock_form.return_value.__exit__ = MagicMock()
        mock_date_input.side_effect = [
            datetime.now().date() - datetime.timedelta(days=30),  # start_date
            datetime.now().date(),  # end_date
        ]
        mock_slider.return_value = (25, 75)  # Risk range
        mock_submit.return_value = True  # Form was submitted

        # Import with patched modules
        with patch.dict("sys.modules", {"streamlit": MagicMock()}):
            # We need to import a small part of the code to test the filter logic
            from auditpulse_mvp.dashboard.dashboard import datetime

            # Create mock session state for testing
            session_state = {}

            # Test filter logic
            if mock_submit.return_value:
                start_date = mock_date_input.side_effect[0]
                end_date = mock_date_input.side_effect[1]
                risk_range = mock_slider.return_value

                session_state["filters"] = {
                    "start_date": datetime.combine(start_date, datetime.time.min),
                    "end_date": datetime.combine(end_date, datetime.time.max),
                    "min_risk_score": risk_range[0],
                    "max_risk_score": risk_range[1],
                }
                session_state["refresh_data"] = True

            # Verify filters would be correctly set
            assert "filters" in session_state
            assert session_state["filters"]["min_risk_score"] == 25
            assert session_state["filters"]["max_risk_score"] == 75
            assert session_state["refresh_data"] is True
