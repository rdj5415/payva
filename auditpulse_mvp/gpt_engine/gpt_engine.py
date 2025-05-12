"""GPT Engine for AuditPulse MVP.

This module implements GPT-4 powered explanations for anomalies.
Features:
- Robust prompting
- Retry with exponential backoff
- Response sanitization
"""
import asyncio
import logging
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import openai
from fastapi import Depends, HTTPException, status
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from auditpulse_mvp.database.base import get_db_session
from auditpulse_mvp.database.models import AnomalyType, Transaction
from auditpulse_mvp.utils.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODEL = "gpt-4"
FALLBACK_MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 3


class GPTEngine:
    """Engine for generating GPT-4 powered explanations for anomalies."""

    def __init__(
        self, 
        db_session: AsyncSession = Depends(get_db_session),
        api_key: Optional[str] = None,
    ):
        """Initialize the GPT Engine.
        
        Args:
            db_session: The database session.
            api_key: Optional API key. If None, uses the settings.
        """
        self.db_session = db_session
        self.client = AsyncOpenAI(
            api_key=api_key or settings.OPENAI_API_KEY.get_secret_value()
            if settings.OPENAI_API_KEY else None
        )
    
    def _build_prompt(
        self, transaction: Transaction, flags: List[Dict]
    ) -> str:
        """Build a prompt for the GPT model.
        
        Args:
            transaction: The transaction to explain.
            flags: The list of flags raised by the rules engine.
            
        Returns:
            str: The prompt for the GPT model.
        """
        # Format transaction details
        txn_details = (
            f"Transaction ID: {transaction.transaction_id}\n"
            f"Date: {transaction.transaction_date.strftime('%Y-%m-%d')}\n"
            f"Amount: ${transaction.amount:,.2f}\n"
            f"Merchant: {transaction.merchant_name}\n"
            f"Category: {transaction.category}\n"
            f"Description: {transaction.description}\n"
            f"Source: {transaction.source.value}\n"
            f"Account: {transaction.source_account_id}\n"
        )
        
        # Format flags
        flags_text = "\n".join([
            f"- {flag['rule_name']}: {flag['description']}" 
            for flag in flags
        ])
        
        # Construct the complete prompt
        prompt = (
            "You are AuditPulse AI, an expert financial auditor assistant. "
            "Your task is to provide a brief, professional explanation of why "
            "the following transaction was flagged as potentially anomalous. "
            "\n\n"
            "TRANSACTION DETAILS:\n"
            f"{txn_details}\n"
            "DETECTED FLAGS:\n"
            f"{flags_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Explain in 1-2 sentences why this transaction was flagged\n"
            "2. Provide 1-2 possible legitimate explanations for the transaction\n"
            "3. Suggest a specific follow-up action an auditor could take\n"
            "4. Write in a neutral, factual tone\n"
            "5. Use concise language (100 words maximum)\n\n"
            "Your response should be structured in 3 short paragraphs corresponding to points 1-3."
        )
        
        return prompt
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, httpx.ReadTimeout)),
        reraise=True,
    )
    async def _call_openai_api(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Call the OpenAI API with a prompt.
        
        Args:
            prompt: The prompt to send to the API.
            model: The model to use.
            temperature: The sampling temperature to use.
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            str: The generated text.
            
        Raises:
            HTTPException: If there's an error calling the API.
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            
            # Extract the generated text
            return response.choices[0].message.content.strip()
            
        except (openai.RateLimitError, openai.APITimeoutError, httpx.ReadTimeout) as exc:
            # These errors will be retried by the retry decorator
            logger.warning(f"OpenAI API temporary error (will retry): {exc}")
            raise
            
        except openai.APIError as exc:
            # Try once with the fallback model if using the primary model
            if model == DEFAULT_MODEL:
                logger.warning(f"Error with {model}, trying fallback model {FALLBACK_MODEL}")
                return await self._call_openai_api(
                    prompt, 
                    model=FALLBACK_MODEL, 
                    temperature=temperature, 
                    max_tokens=max_tokens
                )
            else:
                logger.error(f"OpenAI API error: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Error calling OpenAI API: {str(exc)}",
                )
            
        except Exception as exc:
            logger.error(f"Unexpected error calling OpenAI API: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error calling OpenAI API: {str(exc)}",
            )
    
    def _sanitize_response(self, response: str) -> str:
        """Sanitize the response from the GPT model.
        
        Args:
            response: The response from the model.
            
        Returns:
            str: The sanitized response.
        """
        # Strip any markdown formatting
        sanitized = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        
        # Remove any HTML tags
        sanitized = re.sub(r'<.*?>', '', sanitized)
        
        # Limit to 500 characters if it's too long
        if len(sanitized) > 500:
            sanitized = sanitized[:497] + "..."
        
        return sanitized.strip()
    
    async def generate_explanation(
        self, 
        transaction: Transaction, 
        flags: List[Dict],
        model: str = DEFAULT_MODEL,
    ) -> str:
        """Generate an explanation for a transaction with anomaly flags.
        
        Args:
            transaction: The transaction to explain.
            flags: The list of flags raised by the rules engine.
            model: The model to use. Defaults to DEFAULT_MODEL.
            
        Returns:
            str: The generated explanation.
            
        Raises:
            ValueError: If flags list is empty.
            HTTPException: If there's an error generating the explanation.
        """
        if not flags:
            raise ValueError("Cannot generate explanation without anomaly flags")
        
        try:
            # Build the prompt
            prompt = self._build_prompt(transaction, flags)
            
            # Call the API
            logger.info(f"Generating explanation for transaction {transaction.id}")
            response = await self._call_openai_api(prompt, model=model)
            
            # Sanitize the response
            sanitized = self._sanitize_response(response)
            
            return sanitized
            
        except (openai.RateLimitError, openai.APITimeoutError) as exc:
            logger.error(f"Failed to generate explanation after retries: {exc}")
            return "Unable to generate explanation due to service limitations. Please try again later."
            
        except Exception as exc:
            logger.error(f"Error generating explanation: {exc}")
            if isinstance(exc, HTTPException):
                raise
            return f"Error generating explanation: {str(exc)}"


# Singleton instance for dependency injection
gpt_engine = GPTEngine()


def get_gpt_engine() -> GPTEngine:
    """Dependency function for FastAPI to get the GPT engine.
    
    Returns:
        GPTEngine: The GPT engine instance.
    """
    return gpt_engine 