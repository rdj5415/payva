"""Model management API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from pydantic import BaseModel, Field, validator

from auditpulse_mvp.ml_engine.model_manager import ModelManager
from auditpulse_mvp.database.models import ModelVersion, ModelPerformance
from auditpulse_mvp.api.deps import get_current_user
from auditpulse_mvp.database.models.user import User

router = APIRouter(prefix="/models", tags=["models"])

class ModelVersionCreate(BaseModel):
    """Model version creation request."""
    
    model_type: str = Field(..., description="Type of model (e.g., 'anomaly_detection')")
    model_data: Dict[str, Any] = Field(..., description="Serialized model data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional model metadata")
    is_active: bool = Field(False, description="Whether this version should be active")
    
    @validator("model_type")
    def validate_model_type(cls, v):
        """Validate model type."""
        valid_types = ["anomaly_detection", "transaction_classifier", "risk_scorer"]
        if v not in valid_types:
            raise ValueError(f"Invalid model type. Must be one of: {', '.join(valid_types)}")
        return v
    
class ModelVersionResponse(BaseModel):
    """Model version response."""
    
    id: UUID
    model_type: str
    version: str
    metadata: Dict[str, Any]
    is_active: bool
    created_at: datetime
    activated_at: Optional[datetime]
    deactivated_at: Optional[datetime]
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True
        
class ModelPerformanceCreate(BaseModel):
    """Model performance creation request."""
    
    model_type: str = Field(..., description="Type of model")
    version: str = Field(..., description="Version identifier")
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    dataset_size: int = Field(..., description="Size of evaluation dataset")
    evaluation_time: float = Field(..., description="Time taken for evaluation")
    
    @validator("metrics")
    def validate_metrics(cls, v):
        """Validate metrics format."""
        required_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in required_metrics:
            if metric not in v:
                raise ValueError(f"Required metric '{metric}' missing from metrics")
        return v
    
class ModelPerformanceResponse(BaseModel):
    """Model performance response."""
    
    id: UUID
    model_type: str
    version: str
    metrics: Dict[str, float]
    dataset_size: int
    evaluation_time: float
    recorded_at: datetime
    
    class Config:
        """Pydantic config."""
        
        orm_mode = True
        
class ModelPerformanceSummary(BaseModel):
    """Model performance summary."""
    
    avg_evaluation_time: float
    avg_dataset_size: float
    evaluation_count: int
    latest_metrics: Dict[str, float]

class ModelPredictionRequest(BaseModel):
    """Model prediction request."""
    
    model_type: str = Field(..., description="Type of model")
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    version: Optional[str] = Field(None, description="Specific version to use (defaults to active)")

class ModelPredictionResponse(BaseModel):
    """Model prediction response."""
    
    model_type: str
    version: str
    prediction: Any
    confidence: Optional[float] = None
    execution_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelValidationRequest(BaseModel):
    """Model validation request."""
    
    model_type: str = Field(..., description="Type of model to validate")
    version: str = Field(..., description="Version identifier to validate")
    validation_data: List[Dict[str, Any]] = Field(..., description="Validation dataset")
    ground_truth: List[Any] = Field(..., description="Expected outcomes")

class ModelValidationResponse(BaseModel):
    """Model validation response."""
    
    model_type: str
    version: str
    metrics: Dict[str, float]
    validation_size: int
    execution_time_ms: float
    validation_success: bool
    errors: List[str] = Field(default_factory=list)

@router.post("/versions", response_model=ModelVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_model_version(
    version: ModelVersionCreate,
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> ModelVersion:
    """Create a new model version.
    
    Args:
        version: Model version creation request
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        ModelVersion: Created model version
    """
    try:
        return await model_manager.create_version(
            model_type=version.model_type,
            model_data=version.model_data,
            metadata=version.metadata,
            is_active=version.is_active,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create model version: {str(e)}"
        )
        
@router.get("/versions/{model_type}", response_model=List[ModelVersionResponse])
async def list_model_versions(
    model_type: str = Path(..., description="Type of model"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> List[ModelVersion]:
    """List model versions.
    
    Args:
        model_type: Type of model
        limit: Maximum number of versions to return
        offset: Number of versions to skip
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        List[ModelVersion]: List of model versions
    """
    try:
        return await model_manager.list_versions(
            model_type=model_type,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model versions: {str(e)}"
        )
    
@router.get("/versions/{model_type}/active", response_model=ModelVersionResponse)
async def get_active_version(
    model_type: str = Path(..., description="Type of model"),
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> ModelVersion:
    """Get the currently active model version.
    
    Args:
        model_type: Type of model
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        ModelVersion: Active model version
        
    Raises:
        HTTPException: If no active version found
    """
    version = await model_manager.get_active_version(model_type)
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active version found for model type: {model_type}"
        )
    return version
    
@router.post("/versions/{model_type}/{version}/activate", response_model=ModelVersionResponse)
async def activate_version(
    model_type: str = Path(..., description="Type of model"),
    version: str = Path(..., description="Version identifier"),
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> ModelVersion:
    """Activate a specific model version.
    
    Args:
        model_type: Type of model
        version: Version identifier
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        ModelVersion: Activated model version
        
    Raises:
        HTTPException: If version not found
    """
    try:
        return await model_manager.activate_version(model_type, version)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate model version: {str(e)}"
        )
        
@router.post("/versions/{model_type}/{version}/rollback", response_model=ModelVersionResponse)
async def rollback_version(
    model_type: str = Path(..., description="Type of model"),
    version: str = Path(..., description="Version identifier to rollback to"),
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> ModelVersion:
    """Rollback to a previous model version.
    
    Args:
        model_type: Type of model
        version: Version identifier to rollback to
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        ModelVersion: Rolled back model version
        
    Raises:
        HTTPException: If version not found
    """
    try:
        return await model_manager.rollback_version(model_type, version)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback model version: {str(e)}"
        )
        
@router.post("/performance", response_model=ModelPerformanceResponse, status_code=status.HTTP_201_CREATED)
async def record_performance(
    performance: ModelPerformanceCreate,
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> ModelPerformance:
    """Record model performance metrics.
    
    Args:
        performance: Performance metrics to record
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        ModelPerformance: Recorded performance metrics
    """
    try:
        # Validate version exists
        version_exists = await model_manager.get_version(
            performance.model_type, performance.version
        )
        if not version_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version not found: {performance.model_type} {performance.version}"
            )
        
        return await model_manager.record_performance(
            model_type=performance.model_type,
            version=performance.version,
            metrics=performance.metrics,
            dataset_size=performance.dataset_size,
            evaluation_time=performance.evaluation_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record performance metrics: {str(e)}"
        )
    
@router.get("/performance/{model_type}", response_model=List[ModelPerformanceResponse])
async def get_performance_history(
    model_type: str = Path(..., description="Type of model"),
    version: Optional[str] = Query(None, description="Optional version identifier"),
    limit: int = Query(10, ge=1, le=100),
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> List[ModelPerformance]:
    """Get performance history for a model.
    
    Args:
        model_type: Type of model
        version: Optional version identifier
        limit: Maximum number of records to return
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        List[ModelPerformance]: Performance history
    """
    try:
        return await model_manager.get_performance_history(
            model_type=model_type,
            version=version,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance history: {str(e)}"
        )
    
@router.get("/performance/{model_type}/summary", response_model=ModelPerformanceSummary)
async def get_performance_summary(
    model_type: str = Path(..., description="Type of model"),
    version: Optional[str] = Query(None, description="Optional version identifier"),
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get performance summary for a model.
    
    Args:
        model_type: Type of model
        version: Optional version identifier
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        Dict[str, Any]: Performance summary
    """
    try:
        return await model_manager.get_performance_summary(
            model_type=model_type,
            version=version,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance summary: {str(e)}"
        )

@router.post("/validate", response_model=ModelValidationResponse)
async def validate_model(
    validation: ModelValidationRequest,
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Validate a model version using provided data.
    
    Args:
        validation: Model validation request
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        Dict[str, Any]: Validation results
    """
    try:
        # Check if model version exists
        version = await model_manager.get_version(validation.model_type, validation.version)
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version not found: {validation.model_type} {validation.version}"
            )
        
        # Check validation data
        if len(validation.validation_data) != len(validation.ground_truth):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Validation data and ground truth must have the same length"
            )
        
        if len(validation.validation_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Validation data cannot be empty"
            )
        
        # In a real implementation, this would use the model to make predictions
        # and compare them with ground truth. For MVP, we'll simulate it.
        import time
        import random
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        start_time = time.time()
        # Simulate model predictions
        predictions = [random.choice([0, 1]) for _ in range(len(validation.ground_truth))]
        
        # Calculate metrics
        try:
            metrics = {
                "accuracy": accuracy_score(validation.ground_truth, predictions),
                "precision": precision_score(validation.ground_truth, predictions, zero_division=0),
                "recall": recall_score(validation.ground_truth, predictions, zero_division=0),
                "f1_score": f1_score(validation.ground_truth, predictions, zero_division=0)
            }
        except Exception as e:
            return {
                "model_type": validation.model_type,
                "version": validation.version,
                "metrics": {},
                "validation_size": len(validation.validation_data),
                "execution_time_ms": (time.time() - start_time) * 1000,
                "validation_success": False,
                "errors": [f"Error calculating metrics: {str(e)}"]
            }
        
        execution_time = (time.time() - start_time) * 1000
        
        # Store performance metrics
        await model_manager.record_performance(
            model_type=validation.model_type,
            version=validation.version,
            metrics=metrics,
            dataset_size=len(validation.validation_data),
            evaluation_time=execution_time / 1000  # Convert to seconds
        )
        
        return {
            "model_type": validation.model_type,
            "version": validation.version,
            "metrics": metrics,
            "validation_size": len(validation.validation_data),
            "execution_time_ms": execution_time,
            "validation_success": True,
            "errors": []
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during model validation: {str(e)}"
        )

@router.post("/predict", response_model=ModelPredictionResponse)
async def predict(
    request: ModelPredictionRequest,
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Make a prediction using a model.
    
    Args:
        request: Prediction request
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        Dict[str, Any]: Prediction result
    """
    try:
        # Get model version
        version_obj = None
        if request.version:
            version_obj = await model_manager.get_version(request.model_type, request.version)
            if not version_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model version not found: {request.model_type} {request.version}"
                )
        else:
            version_obj = await model_manager.get_active_version(request.model_type)
            if not version_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No active version found for model type: {request.model_type}"
                )
        
        # In a real implementation, this would use the model to make a prediction
        # For MVP, we'll simulate it
        import time
        import random
        
        start_time = time.time()
        
        # Simulate prediction
        prediction_result = None
        confidence = None
        
        if request.model_type == "anomaly_detection":
            prediction_result = random.choice([0, 1])
            confidence = random.uniform(0.7, 0.99)
        elif request.model_type == "transaction_classifier":
            categories = ["grocery", "entertainment", "utilities", "travel", "dining"]
            prediction_result = random.choice(categories)
            confidence = random.uniform(0.6, 0.95)
        elif request.model_type == "risk_scorer":
            prediction_result = random.uniform(0, 100)
            confidence = random.uniform(0.8, 0.98)
        else:
            prediction_result = "unknown"
            confidence = 0.5
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "model_type": request.model_type,
            "version": version_obj.version,
            "prediction": prediction_result,
            "confidence": confidence,
            "execution_time_ms": execution_time,
            "metadata": {
                "model_id": str(version_obj.id),
                "input_features": list(request.data.keys()),
                "prediction_time": datetime.now().isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@router.get("/health/{model_type}", response_model=Dict[str, Any])
async def model_health_check(
    model_type: str = Path(..., description="Type of model"),
    version: Optional[str] = Query(None, description="Optional version identifier (defaults to active)"),
    model_manager: ModelManager = Depends(),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Check the health of a model.
    
    Args:
        model_type: Type of model
        version: Optional version identifier
        model_manager: Model manager instance
        current_user: Current authenticated user
        
    Returns:
        Dict[str, Any]: Model health status
    """
    try:
        # Get model version
        version_obj = None
        if version:
            version_obj = await model_manager.get_version(model_type, version)
            if not version_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model version not found: {model_type} {version}"
                )
        else:
            version_obj = await model_manager.get_active_version(model_type)
            if not version_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No active version found for model type: {model_type}"
                )
        
        # Get performance metrics
        performance = await model_manager.get_performance_history(
            model_type=model_type,
            version=version_obj.version,
            limit=1
        )
        
        return {
            "status": "healthy",
            "model_type": model_type,
            "version": version_obj.version,
            "is_active": version_obj.is_active,
            "created_at": version_obj.created_at.isoformat(),
            "last_performance_check": performance[0].recorded_at.isoformat() if performance else None,
            "performance_metrics": performance[0].metrics if performance else None,
            "metadata": version_obj.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking model health: {str(e)}"
        ) 