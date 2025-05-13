"""Tests for task API endpoints."""

from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient

from auditpulse_mvp.main import app
from auditpulse_mvp.database.models import TaskStatus
from auditpulse_mvp.tasks.task_manager import TaskPriority

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def mock_task_manager(monkeypatch):
    """Mock task manager."""
    class MockTaskManager:
        def __init__(self):
            self.tasks = {}
            
        async def schedule_task(self, **kwargs):
            task_id = "test-task-id"
            self.tasks[task_id] = {
                "id": task_id,
                "name": kwargs["task_name"],
                "status": TaskStatus.PENDING,
                "priority": kwargs.get("priority", TaskPriority.MEDIUM).name,
                "created_at": datetime.now(),
                "retry_count": 0,
            }
            return task_id
            
        async def get_task_status(self, task_id):
            return self.tasks.get(task_id)
            
        async def get_task_stats(self):
            return {
                "status_counts": {TaskStatus.PENDING: 1},
                "priority_counts": {TaskPriority.MEDIUM.name: 1},
                "avg_execution_time": 0.0,
                "total_tasks": 1,
            }
    
    manager = MockTaskManager()
    monkeypatch.setattr("auditpulse_mvp.api.api_v1.endpoints.tasks.get_task_manager", lambda: manager)
    return manager

def test_create_task(client, mock_task_manager):
    """Test task creation endpoint."""
    response = client.post(
        "/api/v1/tasks",
        json={
            "name": "test_task",
            "priority": TaskPriority.HIGH.name,
            "args": ["arg1"],
            "kwargs": {"kwarg1": "value1"},
        },
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_task"
    assert data["status"] == TaskStatus.PENDING
    assert data["priority"] == TaskPriority.HIGH.name

def test_get_task(client, mock_task_manager):
    """Test get task endpoint."""
    # Create a task first
    task_id = "test-task-id"
    mock_task_manager.tasks[task_id] = {
        "id": task_id,
        "name": "test_task",
        "status": TaskStatus.COMPLETED,
        "priority": TaskPriority.MEDIUM.name,
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "retry_count": 0,
    }
    
    response = client.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == task_id
    assert data["name"] == "test_task"
    assert data["status"] == TaskStatus.COMPLETED

def test_get_task_not_found(client, mock_task_manager):
    """Test get task endpoint with non-existent task."""
    response = client.get("/api/v1/tasks/non-existent")
    assert response.status_code == 404

def test_get_task_stats(client, mock_task_manager):
    """Test get task stats endpoint."""
    response = client.get("/api/v1/tasks/stats")
    assert response.status_code == 200
    data = response.json()
    assert "status_counts" in data
    assert "priority_counts" in data
    assert "avg_execution_time" in data
    assert "total_tasks" in data

def test_create_task_invalid(client, mock_task_manager):
    """Test task creation with invalid data."""
    response = client.post(
        "/api/v1/tasks",
        json={
            "name": "non_existent_task",
            "priority": "INVALID",
        },
    )
    assert response.status_code == 400

def test_create_task_with_schedule(client, mock_task_manager):
    """Test task creation with scheduling."""
    response = client.post(
        "/api/v1/tasks",
        json={
            "name": "test_task",
            "run_at": (datetime.now() + timedelta(hours=1)).isoformat(),
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_task"
    assert data["status"] == TaskStatus.PENDING

def test_create_task_with_interval(client, mock_task_manager):
    """Test task creation with interval."""
    response = client.post(
        "/api/v1/tasks",
        json={
            "name": "test_task",
            "interval": 3600,  # 1 hour
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_task"
    assert data["status"] == TaskStatus.PENDING

def test_create_task_with_cron(client, mock_task_manager):
    """Test task creation with cron expression."""
    response = client.post(
        "/api/v1/tasks",
        json={
            "name": "test_task",
            "cron": "0 * * * *",  # Every hour
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_task"
    assert data["status"] == TaskStatus.PENDING 