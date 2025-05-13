"""Tests for the Risk Explanation Provider.

This module contains tests for the explanation providers that generate
human-readable explanations for risk anomalies.
"""
import pytest
from unittest.mock import Mock, patch

from auditpulse_mvp.risk_engine.explanations import (
    ExplanationRequest,
    RiskExplanationProvider,
    GPTExplanationProvider,
    get_explanation_provider
)


class TestExplanationRequest:
    """Tests for the ExplanationRequest model."""
    
    def test_valid_request(self):
        """Test that a valid request can be created."""
        request = ExplanationRequest(
            anomaly_id=1,
            transaction_amount=1000.0,
            transaction_type="expense",
            merchant_name="Test Merchant",
            transaction_date="2023-06-15",
            ml_score=0.8,
            rule_score=60,
            risk_score=75,
            risk_level="medium",
            risk_factors=["High ML score", "Unusual amount"]
        )
        
        assert request.anomaly_id == 1
        assert request.transaction_amount == 1000.0
        assert request.transaction_type == "expense"
        assert request.ml_score == 0.8
        assert request.risk_level == "medium"
        
    def test_to_dict_method(self):
        """Test the to_dict method."""
        request = ExplanationRequest(
            anomaly_id=1,
            transaction_amount=1000.0,
            transaction_type="expense",
            merchant_name="Test Merchant",
            transaction_date="2023-06-15",
            ml_score=0.8,
            rule_score=60,
            risk_score=75,
            risk_level="medium",
            risk_factors=["High ML score"],
            # Add an optional field
            category="Office Supplies"
        )
        
        result = request.to_dict()
        
        # Check that all required fields are included
        assert result["anomaly_id"] == 1
        assert result["transaction_amount"] == 1000.0
        assert result["ml_score"] == 0.8
        
        # Check that optional fields are included only if not None
        assert "category" in result
        assert result["category"] == "Office Supplies"
        assert "description" not in result  # This optional field was not set


class TestRiskExplanationProvider:
    """Tests for the base RiskExplanationProvider class."""
    
    def test_get_explanation(self):
        """Test getting an explanation for an anomaly."""
        provider = RiskExplanationProvider()
        
        anomaly = {
            "id": 1,
            "transaction_amount": 1000.0,
            "transaction_type": "expense",
            "merchant_name": "Test Merchant",
            "transaction_date": "2023-06-15",
            "ml_score": 0.8,
            "rule_score": 60,
            "risk_score": 75,
            "risk_level": "medium",
            "risk_factors": ["High ML score", "Unusual amount"]
        }
        
        explanation = provider.get_explanation(anomaly)
        
        # Check that the explanation contains relevant information
        assert "medium" in explanation.lower()
        assert "75" in explanation
        assert "1000.00" in explanation
        assert "Test Merchant" in explanation
        assert "2023-06-15" in explanation
    
    def test_prepare_request(self):
        """Test preparing an explanation request from anomaly data."""
        provider = RiskExplanationProvider()
        
        anomaly = {
            "id": 1,
            "transaction_amount": 1000.0,
            "transaction_type": "expense",
            "merchant_name": "Test Merchant",
            "transaction_date": "2023-06-15",
            "ml_score": 0.8,
            "rule_score": 60,
            "risk_score": 75,
            "risk_level": "medium",
            "risk_factors": ["High ML score"]
        }
        
        request = provider._prepare_request(anomaly)
        
        assert isinstance(request, ExplanationRequest)
        assert request.anomaly_id == 1
        assert request.transaction_amount == 1000.0
        assert request.ml_score == 0.8
    
    def test_missing_fields_handling(self):
        """Test handling of missing fields in anomaly data."""
        provider = RiskExplanationProvider()
        
        # Create anomaly with minimal fields
        anomaly = {
            "id": 1,
            "merchant_name": "Test Merchant",
            "risk_level": "high"
        }
        
        # This should not raise an exception
        explanation = provider.get_explanation(anomaly)
        
        # Check that the explanation is generated with default values
        assert explanation
        assert "high" in explanation.lower()
        assert "Test Merchant" in explanation
        assert "0.00" in explanation  # Default amount
    
    def test_get_fallback_explanation(self):
        """Test the fallback explanation when main generation fails."""
        provider = RiskExplanationProvider()
        
        # Mock _prepare_request to raise an exception
        with patch.object(provider, '_prepare_request', side_effect=ValueError("Test error")):
            anomaly = {
                "id": 1,
                "merchant_name": "Test Merchant",
                "transaction_amount": 1000.0
            }
            
            explanation = provider.get_explanation(anomaly)
            
            # Check that fallback explanation is returned
            assert "Test Merchant" in explanation
            assert "1000.00" in explanation
            assert "flagged as unusual" in explanation


class TestGPTExplanationProvider:
    """Tests for the GPT-based explanation provider."""
    
    def test_initialization(self):
        """Test initialization with and without API key."""
        # Test with explicit API key
        provider = GPTExplanationProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        
        # Test without API key (should use settings)
        with patch("auditpulse_mvp.risk_engine.explanations.settings") as mock_settings:
            mock_settings.OPENAI_API_KEY = None
            provider = GPTExplanationProvider()
            assert provider.api_key is None
    
    def test_enhanced_template_explanation(self):
        """Test the enhanced template explanation."""
        provider = GPTExplanationProvider()
        
        request = ExplanationRequest(
            anomaly_id=1,
            transaction_amount=5000.0,
            transaction_type="expense",
            merchant_name="New Vendor",
            transaction_date="2023-06-15",
            ml_score=0.9,
            rule_score=80,
            risk_score=85,
            risk_level="high",
            risk_factors=["High ML score", "Large amount", "New merchant"],
            rule_triggers=["amount_threshold", "new_merchant"]
        )
        
        explanation = provider._enhanced_template_explanation(request)
        
        # Check that the explanation contains enhanced details
        assert "significant risk score of 85/100" in explanation
        assert "following risk factors were identified" in explanation
        assert "machine learning model detected unusual patterns" in explanation
        assert "transaction triggered the following rules" in explanation
        assert "Amount Threshold" in explanation
        assert "New Merchant" in explanation
        assert "Recommendation" in explanation
    
    @patch("auditpulse_mvp.risk_engine.explanations.settings")
    def test_generate_explanation_with_gpt_disabled(self, mock_settings):
        """Test generating an explanation when GPT is disabled."""
        mock_settings.enable_gpt_explanations = False
        
        provider = GPTExplanationProvider()
        
        # Patch the enhanced template method to ensure it's not called
        with patch.object(provider, '_enhanced_template_explanation') as mock_enhanced:
            # Patch the parent class method to track it being called
            with patch.object(RiskExplanationProvider, '_generate_explanation', return_value="Base explanation") as mock_base:
                request = ExplanationRequest(
                    anomaly_id=1,
                    transaction_amount=1000.0,
                    transaction_type="expense",
                    merchant_name="Test Merchant",
                    transaction_date="2023-06-15",
                    ml_score=0.8,
                    rule_score=60,
                    risk_score=75,
                    risk_level="medium",
                    risk_factors=["High ML score"]
                )
                
                explanation = provider._generate_explanation(request)
                
                # Should call parent method
                mock_base.assert_called_once()
                # Should not call enhanced method
                mock_enhanced.assert_not_called()
                assert explanation == "Base explanation"


class TestExplanationProviderSingleton:
    """Tests for the singleton/global explanation provider functionality."""
    
    @patch("auditpulse_mvp.risk_engine.explanations.settings")
    def test_get_explanation_provider_gpt_enabled(self, mock_settings):
        """Test getting the explanation provider with GPT enabled."""
        mock_settings.enable_gpt_explanations = True
        
        # Reset the singleton for testing
        from auditpulse_mvp.risk_engine.explanations import _explanation_provider
        import auditpulse_mvp.risk_engine.explanations
        auditpulse_mvp.risk_engine.explanations._explanation_provider = None
        
        # Get the provider
        provider = get_explanation_provider()
        
        # Check that we got the GPT provider
        assert isinstance(provider, GPTExplanationProvider)
        
        # Check that we get the same instance next time
        provider2 = get_explanation_provider()
        assert provider is provider2
    
    @patch("auditpulse_mvp.risk_engine.explanations.settings")
    def test_get_explanation_provider_gpt_disabled(self, mock_settings):
        """Test getting the explanation provider with GPT disabled."""
        mock_settings.enable_gpt_explanations = False
        
        # Reset the singleton for testing
        from auditpulse_mvp.risk_engine.explanations import _explanation_provider
        import auditpulse_mvp.risk_engine.explanations
        auditpulse_mvp.risk_engine.explanations._explanation_provider = None
        
        # Get the provider
        provider = get_explanation_provider()
        
        # Check that we got the base provider
        assert isinstance(provider, RiskExplanationProvider)
        assert not isinstance(provider, GPTExplanationProvider)
        
        # Check that we get the same instance next time
        provider2 = get_explanation_provider()
        assert provider is provider2 