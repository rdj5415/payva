"""
Load testing for AuditPulse MVP.

This module contains load tests using Locust to simulate multiple
users accessing the system simultaneously.

To run these tests:
1. Install locust: pip install locust
2. Run: locust -f auditpulse_mvp/tests/load_tests.py
3. Open http://localhost:8089 in your browser
"""

import json
import random
from locust import HttpUser, task, between, tag


class AuditPulseUser(HttpUser):
    """
    Simulates a user of the AuditPulse application.
    """
    wait_time = between(1, 5)  # Wait between 1 and 5 seconds between tasks
    
    def on_start(self):
        """
        Setup before tests run - log in and get auth token.
        """
        # Login as test user
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": "test@example.com", 
                "password": "testpassword"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.auth_header = {"Authorization": f"Bearer {self.token}"}
        else:
            # For load testing, create a dummy token if login fails
            self.token = "dummy_token_for_load_testing"
            self.auth_header = {"Authorization": f"Bearer {self.token}"}

    @tag("health")
    @task(10)  # Higher weight = more frequent
    def check_health(self):
        """
        Check the health endpoint.
        """
        self.client.get("/health")

    @tag("public")
    @task(3)
    def view_docs(self):
        """
        View API documentation.
        """
        self.client.get("/api/docs")

    @tag("auth")
    @task(5)
    def get_user_info(self):
        """
        Get current user information.
        """
        self.client.get(
            "/api/v1/users/me",
            headers=self.auth_header
        )

    @tag("transactions")
    @task(7)
    def list_transactions(self):
        """
        List transactions, paginated with random offset.
        """
        offset = random.randint(0, 50)
        self.client.get(
            f"/api/v1/transactions/?limit=10&offset={offset}",
            headers=self.auth_header
        )

    @tag("transactions")
    @task(2)
    def create_transaction(self):
        """
        Create a new transaction.
        """
        amount = round(random.uniform(1.0, 1000.0), 2)
        categories = ["Food", "Travel", "Shopping", "Bills", "Entertainment"]
        
        transaction_data = {
            "amount": amount,
            "date": "2023-01-01T12:00:00",
            "description": f"Load Test Transaction {random.randint(1000, 9999)}",
            "category": random.choice(categories),
            "account_id": "load-test-account"
        }
        
        self.client.post(
            "/api/v1/transactions/",
            json=transaction_data,
            headers=self.auth_header
        )

    @tag("anomalies")
    @task(4)
    def check_anomalies(self):
        """
        Check for anomalies.
        """
        self.client.get(
            "/api/v1/anomalies/?limit=10",
            headers=self.auth_header
        )

    @tag("models")
    @task(1)
    def get_model_info(self):
        """
        Get information about models.
        """
        model_types = ["anomaly_detection", "transaction_classifier"]
        model_type = random.choice(model_types)
        
        self.client.get(
            f"/api/v1/models/{model_type}/versions",
            headers=self.auth_header
        )

    @tag("api")
    @task(3)
    def complex_query(self):
        """
        Perform a more complex query that might hit multiple DB tables.
        """
        # Get transactions from a specific date range
        start_date = "2023-01-01T00:00:00"
        end_date = "2023-01-31T23:59:59"
        
        self.client.get(
            f"/api/v1/transactions/?start_date={start_date}&end_date={end_date}&limit=20",
            headers=self.auth_header
        )
        
        # Then check if any anomalies were detected in that period
        self.client.get(
            f"/api/v1/anomalies/?start_date={start_date}&end_date={end_date}&limit=20",
            headers=self.auth_header
        )


class AuditPulseAdminUser(HttpUser):
    """
    Simulates an admin user of the AuditPulse application.
    Admin users perform different actions and have more privileges.
    """
    wait_time = between(2, 8)  # Wait between 2 and 8 seconds between tasks
    weight = 1  # Lower weight means fewer admin users compared to regular users
    
    def on_start(self):
        """
        Setup before tests run - log in as admin and get auth token.
        """
        # Login as admin user
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "username": "admin@example.com", 
                "password": "adminpassword"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.auth_header = {"Authorization": f"Bearer {self.token}"}
        else:
            # For load testing, create a dummy token if login fails
            self.token = "dummy_admin_token_for_load_testing"
            self.auth_header = {"Authorization": f"Bearer {self.token}"}

    @tag("admin")
    @task(5)
    def list_users(self):
        """
        List all users (admin-only action).
        """
        self.client.get(
            "/api/v1/admin/users",
            headers=self.auth_header
        )

    @tag("admin")
    @task(3)
    def system_health(self):
        """
        Get detailed system health information.
        """
        self.client.get(
            "/api/v1/admin/system-health",
            headers=self.auth_header
        )

    @tag("admin", "models")
    @task(2)
    def model_management(self):
        """
        Perform model management operations.
        """
        model_types = ["anomaly_detection", "transaction_classifier"]
        model_type = random.choice(model_types)
        
        # List model versions
        self.client.get(
            f"/api/v1/models/{model_type}/versions",
            headers=self.auth_header
        )
        
        # Get model performance
        self.client.get(
            f"/api/v1/models/{model_type}/performance",
            headers=self.auth_header
        )

    @tag("admin")
    @task(1)
    def tenant_management(self):
        """
        Perform tenant management operations.
        """
        self.client.get(
            "/api/v1/admin/tenants",
            headers=self.auth_header
        )


if __name__ == "__main__":
    # This allows running with: python -m auditpulse_mvp.tests.load_tests
    # Useful for debugging
    print("Load tests defined. Please use Locust to run these tests.")
    print("Run: locust -f auditpulse_mvp/tests/load_tests.py")
    print("Then open http://localhost:8089 in your browser") 