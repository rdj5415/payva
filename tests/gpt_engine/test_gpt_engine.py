"""Tests for the GPT Engine module.

This module contains tests for the GPT-powered explanation generation,
covering prompt creation, API calls, and response handling.
"""
import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import openai
import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from auditpulse_mvp.database.models import DataSource, Transaction
from auditpulse_mvp.gpt_engine.gpt_engine import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    FALLBACK_MODEL,
    GPTEngine,
    get_gpt_engine,
)


@pytest.fixture
def mock_transaction() -> Transaction:
    """Create a mock transaction for testing."""
    return Transaction(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        transaction_id="test-txn-001",
        source=DataSource.QUICKBOOKS,
        source_account_id="test-account",
        amount=12500.0,
        currency="USD",
        description="Test transaction for equipment purchase",
        category="Equipment",
        merchant_name="Equipment Vendor Inc",
        transaction_date=datetime.datetime.now(),
    )


@pytest.fixture
def mock_flags() -> list[dict]:
    """Create mock flags for testing."""
    return [
        {
            "rule_name": "Large Amount",
            "rule_type": "large_amount",
            "score": 0.75,
            "description": "Transaction amount ($12,500.00) exceeds threshold ($10,000.00)",
            "weight": 1.0,
        },
        {
            "rule_name": "Unapproved Vendor",
            "rule_type": "unapproved_vendor",
            "score": 1.0,
            "description": "Vendor 'Equipment Vendor Inc' is not in the approved vendor list",
            "weight": 1.0,
        },
    ]


@pytest.fixture
def mock_openai_response() -> str:
    """Create a mock OpenAI API response."""
    return (
        "This transaction was flagged due to its large amount ($12,500.00) exceeding the threshold "
        "of $10,000.00 and because 'Equipment Vendor Inc' is not on the approved vendor list. "
        "Both of these factors increase the risk profile of this transaction.\n\n"
        "A legitimate explanation could be that this is a one-time purchase of essential equipment "
        "that required management approval or that Equipment Vendor Inc is a new supplier chosen "
        "for better pricing or product quality but the vendor approval process hasn't been completed yet.\n\n"
        "An auditor should review the purchase order and approval documentation to verify proper "
        "authorization, confirm the receipt of goods, and ensure the vendor was vetted appropriately. "
        "If this is a new vendor relationship, they should check if the vendor approval process is underway."
    )


@pytest.mark.asyncio
async def test_gpt_engine_initialization(db_session):
    """Test GPT Engine initialization."""
    with patch("auditpulse_mvp.gpt_engine.gpt_engine.AsyncOpenAI") as mock_client:
        # Create a custom API key
        api_key = "test-api-key"
        
        # Initialize engine with custom API key
        engine = GPTEngine(db_session, api_key=api_key)
        
        assert engine.db_session == db_session
        mock_client.assert_called_once()
        # Check that API key was passed to client
        assert mock_client.call_args[1]["api_key"] == api_key


@pytest.mark.asyncio
async def test_build_prompt(mock_transaction, mock_flags):
    """Test building a prompt for GPT."""
    engine = GPTEngine()
    
    prompt = engine._build_prompt(mock_transaction, mock_flags)
    
    # Check that prompt contains important elements
    assert "AuditPulse AI" in prompt
    assert mock_transaction.transaction_id in prompt
    assert f"${mock_transaction.amount:,.2f}" in prompt
    assert mock_transaction.merchant_name in prompt
    
    # Check that flags are included
    for flag in mock_flags:
        assert flag["rule_name"] in prompt
        assert flag["description"] in prompt
    
    # Check that instructions are included
    assert "INSTRUCTIONS:" in prompt
    assert "1. Explain" in prompt
    assert "2. Provide" in prompt
    assert "3. Suggest" in prompt


@pytest.mark.asyncio
async def test_call_openai_api_success(mock_transaction, mock_flags, mock_openai_response):
    """Test successful OpenAI API call."""
    engine = GPTEngine()
    
    # Create mock response
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = mock_openai_response
    
    # Mock the client.chat.completions.create method
    engine.client = AsyncMock()
    engine.client.chat.completions.create.return_value = mock_completion
    
    # Call the API
    prompt = engine._build_prompt(mock_transaction, mock_flags)
    response = await engine._call_openai_api(prompt)
    
    assert response == mock_openai_response
    engine.client.chat.completions.create.assert_called_once()
    call_kwargs = engine.client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == DEFAULT_MODEL
    assert call_kwargs["temperature"] == DEFAULT_TEMPERATURE
    assert call_kwargs["max_tokens"] == DEFAULT_MAX_TOKENS


@pytest.mark.asyncio
async def test_call_openai_api_rate_limit_retry():
    """Test OpenAI API rate limit retry behavior."""
    engine = GPTEngine()
    
    # Create mock response for success after retry
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Success after retry"
    
    # Mock the client to fail once with rate limit then succeed
    engine.client = AsyncMock()
    engine.client.chat.completions.create.side_effect = [
        openai.RateLimitError("Rate limit exceeded"),
        mock_completion,
    ]
    
    # Patch tenacity retry to make test faster
    with patch("auditpulse_mvp.gpt_engine.gpt_engine.retry", lambda f, **kwargs: f):
        with pytest.raises(openai.RateLimitError):
            await engine._call_openai_api("Test prompt")
        
        # First call should raise rate limit error for retry
        engine.client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_call_openai_api_fallback_model():
    """Test fallback to GPT-3.5 on API error."""
    engine = GPTEngine()
    
    # Create mock response for success with fallback model
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Success with fallback model"
    
    # Mock the client to fail with primary model, succeed with fallback
    engine.client = AsyncMock()
    # First call with default model fails
    engine.client.chat.completions.create.side_effect = [
        openai.APIError("API Error"),
        mock_completion,  # Second call with fallback model succeeds
    ]
    
    # Call the API - should fallback automatically
    response = await engine._call_openai_api("Test prompt")
    
    assert response == "Success with fallback model"
    assert engine.client.chat.completions.create.call_count == 2
    
    # First call should be with default model
    first_call_kwargs = engine.client.chat.completions.create.call_args_list[0][1]
    assert first_call_kwargs["model"] == DEFAULT_MODEL
    
    # Second call should be with fallback model
    second_call_kwargs = engine.client.chat.completions.create.call_args_list[1][1]
    assert second_call_kwargs["model"] == FALLBACK_MODEL


@pytest.mark.asyncio
async def test_sanitize_response():
    """Test response sanitization."""
    engine = GPTEngine()
    
    # Test with markdown
    markdown_response = (
        "Here's an explanation:\n\n"
        "```python\n"
        "def example():\n"
        "    return 'This should be removed'\n"
        "```\n\n"
        "And some **bold text** that should remain."
    )
    sanitized = engine._sanitize_response(markdown_response)
    assert "```" not in sanitized
    assert "def example" not in sanitized
    assert "And some **bold text** that should remain" in sanitized
    
    # Test with HTML
    html_response = (
        "This contains <b>HTML</b> that should be <a href='removed.html'>removed</a>."
    )
    sanitized = engine._sanitize_response(html_response)
    assert "<b>" not in sanitized
    assert "<a href=" not in sanitized
    assert "This contains HTML that should be removed." in sanitized
    
    # Test with long response
    long_response = "A" * 600
    sanitized = engine._sanitize_response(long_response)
    assert len(sanitized) <= 500
    assert sanitized.endswith("...")


@pytest.mark.asyncio
async def test_generate_explanation_success(mock_transaction, mock_flags, mock_openai_response):
    """Test successful explanation generation."""
    engine = GPTEngine()
    
    with patch.object(engine, "_call_openai_api") as mock_call_api, \
         patch.object(engine, "_sanitize_response", return_value="Sanitized response") as mock_sanitize:
        
        mock_call_api.return_value = mock_openai_response
        
        explanation = await engine.generate_explanation(mock_transaction, mock_flags)
        
        assert explanation == "Sanitized response"
        mock_call_api.assert_called_once()
        mock_sanitize.assert_called_once_with(mock_openai_response)


@pytest.mark.asyncio
async def test_generate_explanation_empty_flags(mock_transaction):
    """Test explanation generation with empty flags."""
    engine = GPTEngine()
    
    with pytest.raises(ValueError) as excinfo:
        await engine.generate_explanation(mock_transaction, [])
    
    assert "Cannot generate explanation without anomaly flags" in str(excinfo.value)


@pytest.mark.asyncio
async def test_generate_explanation_rate_limit(mock_transaction, mock_flags):
    """Test handling rate limit errors during explanation generation."""
    engine = GPTEngine()
    
    with patch.object(engine, "_call_openai_api") as mock_call_api:
        # Simulate persistent rate limit error that isn't resolved by retries
        mock_call_api.side_effect = openai.RateLimitError("Rate limit exceeded")
        
        explanation = await engine.generate_explanation(mock_transaction, mock_flags)
        
        assert "service limitations" in explanation
        mock_call_api.assert_called_once()


@pytest.mark.asyncio
async def test_generate_explanation_http_exception(mock_transaction, mock_flags):
    """Test handling HTTP exceptions during explanation generation."""
    engine = GPTEngine()
    
    http_exception = HTTPException(status_code=500, detail="Internal server error")
    
    with patch.object(engine, "_call_openai_api") as mock_call_api:
        mock_call_api.side_effect = http_exception
        
        # Should re-raise the HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await engine.generate_explanation(mock_transaction, mock_flags)
        
        assert excinfo.value == http_exception
        mock_call_api.assert_called_once()


@pytest.mark.asyncio
async def test_generate_explanation_unexpected_error(mock_transaction, mock_flags):
    """Test handling unexpected errors during explanation generation."""
    engine = GPTEngine()
    
    with patch.object(engine, "_call_openai_api") as mock_call_api:
        mock_call_api.side_effect = RuntimeError("Unexpected error")
        
        explanation = await engine.generate_explanation(mock_transaction, mock_flags)
        
        assert "Error generating explanation" in explanation
        assert "Unexpected error" in explanation
        mock_call_api.assert_called_once()


def test_get_gpt_engine():
    """Test the get_gpt_engine dependency function."""
    engine = get_gpt_engine()
    assert isinstance(engine, GPTEngine) 