"""Test suite for the feedback learning API endpoints.

Tests for the endpoints that handle triggering learning processes and
retrieving feedback learning status.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from auditpulse_mvp.api.deps import get_current_tenant, get_current_user, get_db
from auditpulse_mvp.api.api_v1.api import api_router
from auditpulse_mvp.learning.scheduler import FeedbackLearningScheduler


@pytest.fixture
def tenant_id():
    """Generate a test tenant ID."""
    return str(uuid4())


@pytest.fixture
def user_id():
    """Generate a test user ID."""
    return str(uuid4())


@pytest.fixture
def app(tenant_id, user_id):
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(api_router)
    
    # Override the dependency for database session
    async def override_get_db():
        try:
            db = AsyncMock()
            yield db
        finally:
            pass
    
    # Override the dependency for current user
    async def override_get_current_user():
        return {"id": user_id, "email": "test@example.com", "is_active": True}
    
    # Override the dependency for current tenant
    async def override_get_current_tenant():
        return {"id": tenant_id, "name": "Test Tenant"}
    
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_current_tenant] = override_get_current_tenant
    
    return app


@pytest.fixture
def client(app):
    """Create a TestClient for the app."""
    return TestClient(app)


@pytest.mark.asyncio
class TestFeedbackLearningEndpoints:
    """Test the feedback learning API endpoints."""
    
    async def test_trigger_immediate_learning(self, app, client, tenant_id):
        """Test triggering immediate learning for a tenant."""
        # Mock the FeedbackLearningScheduler
        mock_scheduler = AsyncMock()
        mock_scheduler.run_immediate_learning.return_value = {
            "status": "success",
            "tenant_id": tenant_id,
            "feedback_analysis": {"total": 7, "confirm": 4, "dismiss": 3},
            "config_updates": {"sensitivity_before": 0.5, "sensitivity_after": 0.6},
            "retraining_status": {"status": "success"}
        }
        
        # Patch the get_feedback_learning_scheduler function
        with patch('auditpulse_mvp.api.api_v1.endpoints.learning.get_feedback_learning_scheduler') as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            
            # Make a request to trigger immediate learning
            response = client.post(f"/api/v1/learning/trigger/{tenant_id}")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["tenant_id"] == tenant_id
            assert "feedback_analysis" in data
            assert "config_updates" in data
            assert "retraining_status" in data
            
            # Verify that the scheduler was called
            mock_scheduler.run_immediate_learning.assert_called_once_with(tenant_id)
    
    async def test_trigger_all_tenants_learning(self, app, client):
        """Test triggering learning for all tenants."""
        # Mock the FeedbackLearningScheduler
        mock_scheduler = AsyncMock()
        mock_scheduler.run_immediate_learning.return_value = {
            "status": "completed",
            "total_tenants": 2,
            "results": [
                {
                    "tenant_id": "tenant1",
                    "status": "success",
                    "feedback_analysis": {"total": 7, "confirm": 4, "dismiss": 3}
                },
                {
                    "tenant_id": "tenant2",
                    "status": "success",
                    "feedback_analysis": {"total": 5, "confirm": 3, "dismiss": 2}
                }
            ]
        }
        
        # Patch the get_feedback_learning_scheduler function
        with patch('auditpulse_mvp.api.api_v1.endpoints.learning.get_feedback_learning_scheduler') as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            
            # Make a request to trigger learning for all tenants
            response = client.post("/api/v1/learning/trigger")
            
            # Verify the response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["total_tenants"] == 2
            assert len(data["results"]) == 2
            
            # Verify that the scheduler was called
            mock_scheduler.run_immediate_learning.assert_called_once_with()
    
    async def test_get_learning_status(self, app, client, tenant_id):
        """Test getting the learning status for a tenant."""
        # Mock the database session
        mock_db = AsyncMock()
        
        # Mock the SensitivityConfig query result
        mock_config = MagicMock()
        mock_config.sensitivity = 0.6
        mock_config.last_updated = "2023-05-10T15:30:00"
        
        # Mock the query result
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_config
        mock_db.execute.return_value = mock_result
        
        # Override the get_db dependency
        async def override_get_db():
            yield mock_db
        
        app.dependency_overrides[get_db] = override_get_db
        
        # Make a request to get the learning status
        response = client.get(f"/api/v1/learning/status/{tenant_id}")
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == tenant_id
        assert "sensitivity_configs" in data
        assert len(data["sensitivity_configs"]) == 1
        assert data["sensitivity_configs"][0]["sensitivity"] == 0.6
        assert data["sensitivity_configs"][0]["last_updated"] == "2023-05-10T15:30:00"
    
    async def test_get_learning_status_no_config(self, app, client, tenant_id):
        """Test getting the learning status when no config exists."""
        # Mock the database session
        mock_db = AsyncMock()
        
        # Mock the query result with no config found
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute.return_value = mock_result
        
        # Override the get_db dependency
        async def override_get_db():
            yield mock_db
        
        app.dependency_overrides[get_db] = override_get_db
        
        # Make a request to get the learning status
        response = client.get(f"/api/v1/learning/status/{tenant_id}")
        
        # Verify the response
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == f"No sensitivity configurations found for tenant {tenant_id}"


@pytest.mark.asyncio
class TestFeedbackLearningAsyncClient:
    """Test the feedback learning API endpoints using AsyncClient."""
    
    async def test_trigger_immediate_learning_async(self, app, tenant_id):
        """Test triggering immediate learning using an async client."""
        # Mock the FeedbackLearningScheduler
        mock_scheduler = AsyncMock()
        mock_scheduler.run_immediate_learning.return_value = {
            "status": "success",
            "tenant_id": tenant_id,
            "feedback_analysis": {"total": 7, "confirm": 4, "dismiss": 3},
            "config_updates": {"sensitivity_before": 0.5, "sensitivity_after": 0.6},
            "retraining_status": {"status": "success"}
        }
        
        # Patch the get_feedback_learning_scheduler function
        with patch('auditpulse_mvp.api.api_v1.endpoints.learning.get_feedback_learning_scheduler') as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            
            # Make a request to trigger immediate learning
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post(f"/api/v1/learning/trigger/{tenant_id}")
                
                # Verify the response
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["tenant_id"] == tenant_id
                
                # Verify that the scheduler was called
                mock_scheduler.run_immediate_learning.assert_called_once_with(tenant_id)
    
    async def test_handling_learning_errors(self, app, tenant_id):
        """Test handling errors during learning."""
        # Mock the FeedbackLearningScheduler to raise an exception
        mock_scheduler = AsyncMock()
        mock_scheduler.run_immediate_learning.side_effect = Exception("Test learning error")
        
        # Patch the get_feedback_learning_scheduler function
        with patch('auditpulse_mvp.api.api_v1.endpoints.learning.get_feedback_learning_scheduler') as mock_get_scheduler:
            mock_get_scheduler.return_value = mock_scheduler
            
            # Make a request to trigger immediate learning
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post(f"/api/v1/learning/trigger/{tenant_id}")
                
                # Verify the response
                assert response.status_code == 500
                data = response.json()
                assert data["detail"].startswith("Error triggering learning")


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_tenant_id():
    """Generate a test tenant ID."""
    return uuid.uuid4()


@pytest.fixture
def mock_current_tenant(mock_tenant_id):
    """Create a mock current tenant."""
    tenant = MagicMock()
    tenant.id = mock_tenant_id
    return tenant


@pytest.mark.asyncio
class TestLearningEndpoints:
    """Test cases for feedback learning API endpoints."""
    
    async def test_trigger_learning_now_success(self, mock_db, mock_current_tenant):
        """Test successfully triggering immediate learning."""
        # Mock the feedback learner
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.get_feedback_learner") as mock_get_learner:
            mock_learner = AsyncMock()
            mock_result = {
                "status": "success",
                "processed_count": 25,
                "false_positive_rate": 0.4,
                "true_positive_rate": 0.6,
                "config_updated": True,
                "model_retrained": True,
            }
            mock_learner.process_recent_feedback.return_value = mock_result
            mock_get_learner.return_value = mock_learner
            
            # Call the endpoint function
            response = await trigger_learning_now(
                db=mock_db,
                current_tenant=mock_current_tenant,
            )
            
            # Verify the response
            assert response["status"] == "success"
            assert response["processed_count"] == 25
            assert response["false_positive_rate"] == 0.4
            assert response["config_updated"] is True
            
            # Verify that the learner was called correctly
            mock_learner.process_recent_feedback.assert_called_once_with(
                mock_current_tenant.id, days=30
            )
    
    async def test_trigger_learning_no_feedback(self, mock_db, mock_current_tenant):
        """Test triggering learning when no feedback is available."""
        # Mock the feedback learner
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.get_feedback_learner") as mock_get_learner:
            mock_learner = AsyncMock()
            mock_result = {
                "status": "skipped",
                "reason": "No recent feedback found",
                "processed_count": 0,
            }
            mock_learner.process_recent_feedback.return_value = mock_result
            mock_get_learner.return_value = mock_learner
            
            # Call the endpoint function
            response = await trigger_learning_now(
                db=mock_db,
                current_tenant=mock_current_tenant,
            )
            
            # Verify the response
            assert response["status"] == "skipped"
            assert response["reason"] == "No recent feedback found"
            assert response["processed_count"] == 0
    
    async def test_trigger_learning_error(self, mock_db, mock_current_tenant):
        """Test error handling when triggering learning."""
        # Mock the feedback learner
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.get_feedback_learner") as mock_get_learner:
            mock_learner = AsyncMock()
            mock_learner.process_recent_feedback.side_effect = Exception("Test error")
            mock_get_learner.return_value = mock_learner
            
            # Call the endpoint function
            with pytest.raises(HTTPException) as excinfo:
                await trigger_learning_now(
                    db=mock_db,
                    current_tenant=mock_current_tenant,
                )
            
            # Verify the error response
            assert excinfo.value.status_code == 500
            assert "Failed to trigger feedback learning" in str(excinfo.value.detail)
    
    async def test_get_learning_status(self, mock_db, mock_current_tenant):
        """Test getting learning status."""
        # Mock the get_tenant_configuration function
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.get_tenant_configuration") as mock_get_config:
            mock_config = {
                "last_learning_run": {
                    "timestamp": "2023-08-15T10:30:00Z",
                    "status": "success",
                    "processed_count": 20,
                    "rules_updated": 2,
                    "models_updated": 1,
                },
                "scheduled_learning": {
                    "enabled": True,
                    "frequency": "daily",
                    "hour": 2,
                    "minute": 0,
                    "next_run": "2023-08-16T02:00:00Z",
                },
            }
            mock_get_config.return_value = mock_config
            
            # Call the endpoint function
            response = await get_learning_status(
                db=mock_db,
                current_tenant=mock_current_tenant,
            )
            
            # Verify the response
            assert response["last_learning_run"]["status"] == "success"
            assert response["last_learning_run"]["processed_count"] == 20
            assert response["scheduled_learning"]["enabled"] is True
            assert response["scheduled_learning"]["hour"] == 2
    
    async def test_get_learning_status_no_config(self, mock_db, mock_current_tenant):
        """Test getting learning status when no configuration exists."""
        # Mock the get_tenant_configuration function
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.get_tenant_configuration") as mock_get_config:
            mock_get_config.return_value = None
            
            # Call the endpoint function
            response = await get_learning_status(
                db=mock_db,
                current_tenant=mock_current_tenant,
            )
            
            # Verify the response
            assert response["last_learning_run"] is None
            assert response["scheduled_learning"]["enabled"] is True  # Default value
            assert response["scheduled_learning"]["frequency"] == "daily"
    
    async def test_get_feedback_statistics(self, mock_db, mock_current_tenant):
        """Test getting feedback statistics."""
        # Mock the get_learning_statistics function
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.get_learning_statistics") as mock_get_stats:
            mock_stats = {
                "total_feedback": 50,
                "rules_based": {
                    "false_positive": 15,
                    "true_positive": 10,
                    "needs_investigation": 5,
                    "total": 30,
                    "false_positive_rate": 0.5,
                    "true_positive_rate": 0.33,
                },
                "ml_based": {
                    "false_positive": 5,
                    "true_positive": 15,
                    "needs_investigation": 0,
                    "total": 20,
                    "false_positive_rate": 0.25,
                    "true_positive_rate": 0.75,
                },
                "learning_adjustments": {
                    "rules_adjusted": 3,
                    "ml_models_adjusted": 1,
                    "last_adjustment_date": "2023-08-15T10:30:00Z",
                },
            }
            mock_get_stats.return_value = mock_stats
            
            # Call the endpoint function
            response = await get_feedback_statistics(
                db=mock_db,
                current_tenant=mock_current_tenant,
                days_lookback=90,
            )
            
            # Verify the response
            assert response["total_feedback"] == 50
            assert response["rules_based"]["false_positive"] == 15
            assert response["ml_based"]["true_positive_rate"] == 0.75
            assert response["learning_adjustments"]["rules_adjusted"] == 3
    
    async def test_update_learning_schedule(self, mock_db, mock_current_tenant):
        """Test updating the learning schedule."""
        # Create update data
        update_data = LearningScheduleUpdate(
            enabled=True,
            hour=4,
            minute=30,
        )
        
        # Mock the update_tenant_configuration function
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.update_tenant_configuration") as mock_update_config:
            mock_update_config.return_value = True
            
            # Call the endpoint function
            response = await update_learning_schedule(
                update_data=update_data,
                db=mock_db,
                current_tenant=mock_current_tenant,
            )
            
            # Verify the response
            assert response["status"] == "success"
            assert response["update"] == update_data.dict()
            
            # Verify that update_tenant_configuration was called correctly
            mock_update_config.assert_called_once()
            call_args = mock_update_config.call_args[0]
            assert call_args[0] == mock_db  # db
            assert call_args[1] == mock_current_tenant.id  # tenant_id
            assert call_args[2] == "learning_schedule"  # key
            assert call_args[3]["enabled"] is True
            assert call_args[3]["hour"] == 4
            assert call_args[3]["minute"] == 30
    
    async def test_update_learning_schedule_failure(self, mock_db, mock_current_tenant):
        """Test error handling when updating the learning schedule."""
        # Create update data
        update_data = LearningScheduleUpdate(
            enabled=False,
            hour=4,
            minute=30,
        )
        
        # Mock the update_tenant_configuration function
        with patch("auditpulse_mvp.api.api_v1.endpoints.learning.update_tenant_configuration") as mock_update_config:
            mock_update_config.side_effect = Exception("Failed to update configuration")
            
            # Call the endpoint function
            with pytest.raises(HTTPException) as excinfo:
                await update_learning_schedule(
                    update_data=update_data,
                    db=mock_db,
                    current_tenant=mock_current_tenant,
                )
            
            # Verify the error response
            assert excinfo.value.status_code == 500
            assert "Failed to update learning schedule" in str(excinfo.value.detail) 