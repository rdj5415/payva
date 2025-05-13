"""Task management API endpoints."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from auditpulse_mvp.api.api_v1.auth import get_current_superuser
from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import TaskStatus
from auditpulse_mvp.tasks.task_manager import TaskManager, TaskPriority
from auditpulse_mvp.utils.settings import get_settings

router = APIRouter(prefix="/tasks", tags=["Tasks"])


# Pydantic models
class TaskCreate(BaseModel):
    """Task creation request."""

    name: str = Field(..., description="Name of the task to schedule")
    priority: TaskPriority = Field(TaskPriority.MEDIUM, description="Task priority")
    args: Optional[List[Any]] = Field(None, description="Positional arguments")
    kwargs: Optional[Dict[str, Any]] = Field(None, description="Keyword arguments")
    run_at: Optional[datetime] = Field(None, description="Specific time to run")
    interval: Optional[int] = Field(None, description="Interval in seconds")
    cron: Optional[str] = Field(None, description="Cron expression")


class TaskResponse(BaseModel):
    """Task response."""

    id: str
    name: str
    status: TaskStatus
    priority: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    error: Optional[str]
    result: Optional[Dict[str, Any]]


class TaskStats(BaseModel):
    """Task statistics."""

    status_counts: Dict[str, int]
    priority_counts: Dict[str, int]
    avg_execution_time: float
    total_tasks: int


# Dependencies
async def get_task_manager(
    db=Depends(get_db_session),
    settings=Depends(get_settings),
) -> TaskManager:
    """Get task manager instance."""
    return TaskManager(db, settings)


# Endpoints
@router.post(
    "",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_current_superuser)],
)
async def create_task(
    task: TaskCreate,
    task_manager: TaskManager = Depends(get_task_manager),
) -> TaskResponse:
    """Schedule a new task.

    Args:
        task: Task creation request
        task_manager: Task manager instance

    Returns:
        TaskResponse: Created task information

    Raises:
        HTTPException: If task creation fails
    """
    try:
        # Convert interval to timedelta if provided
        interval = timedelta(seconds=task.interval) if task.interval else None

        # Schedule task
        task_id = await task_manager.schedule_task(
            task_name=task.name,
            priority=task.priority,
            args=task.args,
            kwargs=task.kwargs,
            run_at=task.run_at,
            interval=interval,
            cron=task.cron,
        )

        # Get task status
        task_info = await task_manager.get_task_status(task_id)
        if not task_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create task",
            )

        return TaskResponse(**task_info)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}",
        )


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    dependencies=[Depends(get_current_superuser)],
)
async def get_task(
    task_id: str = Path(..., description="Task ID"),
    task_manager: TaskManager = Depends(get_task_manager),
) -> TaskResponse:
    """Get task status.

    Args:
        task_id: Task ID
        task_manager: Task manager instance

    Returns:
        TaskResponse: Task information

    Raises:
        HTTPException: If task not found
    """
    task_info = await task_manager.get_task_status(task_id)
    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    return TaskResponse(**task_info)


@router.get(
    "/stats",
    response_model=TaskStats,
    dependencies=[Depends(get_current_superuser)],
)
async def get_task_stats(
    task_manager: TaskManager = Depends(get_task_manager),
) -> TaskStats:
    """Get task execution statistics.

    Args:
        task_manager: Task manager instance

    Returns:
        TaskStats: Task statistics
    """
    stats = await task_manager.get_task_stats()
    return TaskStats(**stats)
