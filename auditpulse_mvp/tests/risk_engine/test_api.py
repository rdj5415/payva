"""Test suite for the Risk Engine API endpoints.

This module tests the API endpoints for configuring risk weights and sensitivity.
"""
import datetime
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from auditpulse_mvp.risk_engine.api import (
    router as risk_router,
    UpdateWeightsRequest,
    UpdateSensitivityRequest,
    RescoringResponse,
)
from auditpulse_mvp.risk_engine.risk_engine import (
    RiskConfig,
    RiskEngine,
    RiskWeights,
    RiskSensitivity,
)


@pytest.fixture
def mock_tenant_id():
    """Create a mock tenant ID."""
    return uuid.uuid4()


@pytest.fixture
def mock_risk_engine():
    """Create a mock risk engine."""
    engine = AsyncMock(spec=RiskEngine)
    
    # Mock get_tenant_config
    mock_config = MagicMock(spec=RiskConfig)
    mock_config.tenant_id = uuid.uuid4()
    mock_config.weights = MagicMock(spec=RiskWeights)
    mock_config.weights.ml_score = 0.4
    mock_config.weights.rule_score = 0.5
    mock_config.weights.historical = 0.1
    mock_config.sensitivity = MagicMock(spec=RiskSensitivity)
    mock_config.sensitivity.threshold = 50
    mock_config.sensitivity.min_confidence = 0.6
    mock_config.sensitivity.ml_threshold = 0.7
    mock_config.dict = lambda: {
        "tenant_id": str(mock_config.tenant_id),
        "weights": {
            "ml_score": 0.4,
            "rule_score": 0.5,
            "historical": 0.1,
            "rule_types": {
                "large_amount": 1.0,
                "unusual_vendor": 0.8,
            },
        },
        "sensitivity": {
            "threshold": 50,
            "min_confidence": 0.6,
            "ml_threshold": 0.7,
        },
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    
    engine.get_tenant_config.return_value = mock_config
    engine.update_tenant_config.return_value = mock_config
    
    # Mock rescore_anomalies
    rescore_result = {
        "total_anomalies": 10,
        "updated_anomalies": 8,
        "newly_flagged": 3,
        "newly_unflagged": 2,
        "tenant_id": "test-tenant-id",
        "date_range": "2023-01-01 - 2023-02-01",
    }
    engine.rescore_anomalies.return_value = rescore_result
    
    return engine


@pytest.fixture
def app(mock_risk_engine):
    """Create a FastAPI app with the risk router."""
    app = FastAPI()
    
    # Override dependencies
    def get_mock_risk_engine():
        return mock_risk_engine
    
    app.dependency_overrides = {
        "auditpulse_mvp.risk_engine.api.get_risk_engine": get_mock_risk_engine,
    }
    
    app.include_router(risk_router)
    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return TestClient(app)


class TestRiskAPI:
    """Test risk engine API endpoints."""
    
    def test_get_risk_config(self, client, mock_tenant_id, mock_risk_engine):
        """Test getting risk configuration for a tenant."""
        # Make request
        response = client.get(f"/risk/config/{mock_tenant_id}")
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check data
        assert "tenant_id" in data
        assert "weights" in data
        assert "sensitivity" in data
        assert data["weights"]["ml_score"] == 0.4
        assert data["weights"]["rule_score"] == 0.5
        assert data["sensitivity"]["threshold"] == 50
        
        # Check that get_tenant_config was called
        mock_risk_engine.get_tenant_config.assert_called_once_with(mock_tenant_id)
    
    def test_update_risk_weights(self, client, mock_tenant_id, mock_risk_engine):
        """Test updating risk weights."""
        # Prepare request data
        weights = {
            "ml_score": 0.3,
            "rule_score": 0.6,
            "historical": 0.1,
            "rule_types": {
                "large_amount": 1.0,
                "unusual_vendor": 0.8,
            },
        }
        request_data = {"weights": weights}
        
        # Make request
        response = client.put(
            f"/risk/config/{mock_tenant_id}/weights",
            json=request_data,
        )
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check data
        assert "tenant_id" in data
        assert "weights" in data
        assert "sensitivity" in data
        
        # Check that update_tenant_config was called
        mock_risk_engine.update_tenant_config.assert_called_once()
        call_args = mock_risk_engine.update_tenant_config.call_args[0]
        assert call_args[0] == mock_tenant_id
        assert "weights" in call_args[1]
        assert call_args[1]["weights"] == weights
    
    def test_update_risk_sensitivity(self, client, mock_tenant_id, mock_risk_engine):
        """Test updating risk sensitivity."""
        # Prepare request data
        sensitivity = {
            "threshold": 60,
            "min_confidence": 0.7,
            "ml_threshold": 0.8,
        }
        request_data = {"sensitivity": sensitivity}
        
        # Make request
        response = client.put(
            f"/risk/config/{mock_tenant_id}/sensitivity",
            json=request_data,
        )
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check data
        assert "tenant_id" in data
        assert "weights" in data
        assert "sensitivity" in data
        
        # Check that update_tenant_config was called
        mock_risk_engine.update_tenant_config.assert_called_once()
        call_args = mock_risk_engine.update_tenant_config.call_args[0]
        assert call_args[0] == mock_tenant_id
        assert "sensitivity" in call_args[1]
        assert call_args[1]["sensitivity"] == sensitivity
    
    def test_rescore_anomalies(self, client, mock_tenant_id, mock_risk_engine):
        """Test rescoring anomalies."""
        # Make request
        response = client.post(
            f"/risk/rescore/{mock_tenant_id}",
            json={"days": 30},
        )
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check data
        assert "total_anomalies" in data
        assert "updated_anomalies" in data
        assert "newly_flagged" in data
        assert "newly_unflagged" in data
        assert "tenant_id" in data
        assert "date_range" in data
        
        # Check that rescore_anomalies was called
        mock_risk_engine.rescore_anomalies.assert_called_once_with(mock_tenant_id, 30)
    
    def test_error_handling(self, client, mock_tenant_id, mock_risk_engine):
        """Test error handling in API endpoints."""
        # Set up mock to raise exception
        mock_risk_engine.get_tenant_config.side_effect = Exception("Test error")
        
        # Make request
        response = client.get(f"/risk/config/{mock_tenant_id}")
        
        # Check response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "detail" in data
        assert "Error retrieving risk configuration" in data["detail"] 