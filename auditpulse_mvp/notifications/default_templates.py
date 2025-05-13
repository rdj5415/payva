"""Default notification templates for the AuditPulse system."""

import logging
from typing import Dict, Any, List
from auditpulse_mvp.notifications.templates import TemplateManager
from auditpulse_mvp.utils.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Default notification templates
DEFAULT_TEMPLATES = [
    # Anomaly detection notifications
    {
        "template_id": "anomaly_detected",
        "name": "Anomaly Detected",
        "subject": "Financial Anomaly Detected",
        "body": """
Dear {{ user_name }},

Our anomaly detection system has identified unusual activity in your financial transactions.

Transaction details:
- Date: {{ transaction_date }}
- Amount: {{ transaction_amount }}
- Description: {{ transaction_description }}
- Account: {{ account_name }}

Anomaly score: {{ anomaly_score }}
Reason: {{ anomaly_reason }}

This transaction was flagged because it differs significantly from your usual spending patterns. 
If this transaction is legitimate, no action is needed. If you don't recognize this transaction, 
please contact your financial institution immediately.

You can review the transaction details in your AuditPulse dashboard:
{{ dashboard_url }}

Best regards,
The AuditPulse Team
        """,
        "html_body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 5px;">
    <h2 style="color: #d9534f;">Financial Anomaly Detected</h2>
    <p>Dear {{ user_name }},</p>
    
    <p>Our anomaly detection system has identified <strong>unusual activity</strong> in your financial transactions.</p>
    
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0;">Transaction details:</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li><strong>Date:</strong> {{ transaction_date }}</li>
            <li><strong>Amount:</strong> {{ transaction_amount }}</li>
            <li><strong>Description:</strong> {{ transaction_description }}</li>
            <li><strong>Account:</strong> {{ account_name }}</li>
        </ul>
        <p><strong>Anomaly score:</strong> {{ anomaly_score }}</p>
        <p><strong>Reason:</strong> {{ anomaly_reason }}</p>
    </div>
    
    <p>This transaction was flagged because it differs significantly from your usual spending patterns.</p>
    <p>If this transaction is legitimate, no action is needed. If you don't recognize this transaction, please contact your financial institution immediately.</p>
    
    <p><a href="{{ dashboard_url }}" style="display: inline-block; background-color: #5cb85c; color: white; padding: 10px 15px; text-decoration: none; border-radius: 3px; margin-top: 10px;">Review in Dashboard</a></p>
    
    <p>Best regards,<br>The AuditPulse Team</p>
</div>
        """,
        "description": "Notification sent when an anomaly is detected in a financial transaction",
        "placeholders": {
            "user_name": "User's full name",
            "transaction_date": "Date of the transaction",
            "transaction_amount": "Amount of the transaction",
            "transaction_description": "Description of the transaction",
            "account_name": "Name of the account",
            "anomaly_score": "Anomaly score from the model",
            "anomaly_reason": "Reason why the transaction was flagged",
            "dashboard_url": "URL to the anomaly details in the dashboard",
        },
    },
    # Model performance notification
    {
        "template_id": "model_performance_report",
        "name": "Model Performance Report",
        "subject": "Model Performance Report: {{ model_name }}",
        "body": """
Dear {{ user_name }},

Here's the latest performance report for model "{{ model_name }}".

Performance metrics:
- Accuracy: {{ accuracy }}
- Precision: {{ precision }}
- Recall: {{ recall }}
- F1 Score: {{ f1_score }}
- ROC AUC: {{ roc_auc }}

Time period: {{ start_date }} to {{ end_date }}
Total transactions analyzed: {{ total_transactions }}
Anomalies detected: {{ anomalies_detected }}
False positive rate: {{ false_positive_rate }}

{% if performance_change > 0 %}
The model performance has improved by {{ performance_change }}% compared to the previous period.
{% elif performance_change < 0 %}
The model performance has decreased by {{ performance_change|abs }}% compared to the previous period.
{% else %}
The model performance has remained stable compared to the previous period.
{% endif %}

To view the full performance report, visit:
{{ report_url }}

Best regards,
The AuditPulse Team
        """,
        "html_body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 5px;">
    <h2 style="color: #428bca;">Model Performance Report: {{ model_name }}</h2>
    <p>Dear {{ user_name }},</p>
    
    <p>Here's the latest performance report for model "{{ model_name }}".</p>
    
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0;">Performance metrics:</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li><strong>Accuracy:</strong> {{ accuracy }}</li>
            <li><strong>Precision:</strong> {{ precision }}</li>
            <li><strong>Recall:</strong> {{ recall }}</li>
            <li><strong>F1 Score:</strong> {{ f1_score }}</li>
            <li><strong>ROC AUC:</strong> {{ roc_auc }}</li>
        </ul>
        
        <p><strong>Time period:</strong> {{ start_date }} to {{ end_date }}</p>
        <p><strong>Total transactions analyzed:</strong> {{ total_transactions }}</p>
        <p><strong>Anomalies detected:</strong> {{ anomalies_detected }}</p>
        <p><strong>False positive rate:</strong> {{ false_positive_rate }}</p>
    </div>
    
    <div style="margin: 20px 0; padding: 10px 15px; border-radius: 4px; 
                {% if performance_change > 0 %}
                background-color: #dff0d8; color: #3c763d;
                {% elif performance_change < 0 %}
                background-color: #f2dede; color: #a94442;
                {% else %}
                background-color: #d9edf7; color: #31708f;
                {% endif %}">
        {% if performance_change > 0 %}
        <p>The model performance has improved by <strong>{{ performance_change }}%</strong> compared to the previous period.</p>
        {% elif performance_change < 0 %}
        <p>The model performance has decreased by <strong>{{ performance_change|abs }}%</strong> compared to the previous period.</p>
        {% else %}
        <p>The model performance has remained stable compared to the previous period.</p>
        {% endif %}
    </div>
    
    <p><a href="{{ report_url }}" style="display: inline-block; background-color: #428bca; color: white; padding: 10px 15px; text-decoration: none; border-radius: 3px; margin-top: 10px;">View Full Report</a></p>
    
    <p>Best regards,<br>The AuditPulse Team</p>
</div>
        """,
        "description": "Notification sent with model performance reports",
        "placeholders": {
            "user_name": "User's full name",
            "model_name": "Name of the model",
            "accuracy": "Accuracy metric",
            "precision": "Precision metric",
            "recall": "Recall metric",
            "f1_score": "F1 score metric",
            "roc_auc": "ROC AUC metric",
            "start_date": "Start date of the performance period",
            "end_date": "End date of the performance period",
            "total_transactions": "Total transactions analyzed",
            "anomalies_detected": "Number of anomalies detected",
            "false_positive_rate": "False positive rate",
            "performance_change": "Change in performance compared to previous period",
            "report_url": "URL to the full performance report",
        },
    },
    # System alert notification
    {
        "template_id": "system_alert",
        "name": "System Alert",
        "subject": "System Alert: {{ alert_type }}",
        "body": """
ALERT: {{ alert_title }}

Severity: {{ severity }}
Time: {{ alert_time }}
Component: {{ component }}

Description:
{{ alert_description }}

{% if actions %}
Recommended Actions:
{% for action in actions %}
- {{ action }}
{% endfor %}
{% endif %}

Alert ID: {{ alert_id }}

--
AuditPulse System Monitoring
        """,
        "html_body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 5px;">
    <h2 style="color: 
                {% if severity == 'critical' %}
                #d9534f;
                {% elif severity == 'high' %}
                #f0ad4e;
                {% elif severity == 'medium' %}
                #5bc0de;
                {% else %}
                #5cb85c;
                {% endif %}">
        ALERT: {{ alert_title }}
    </h2>
    
    <div style="background-color: 
                {% if severity == 'critical' %}
                #f2dede;
                {% elif severity == 'high' %}
                #fcf8e3;
                {% elif severity == 'medium' %}
                #d9edf7;
                {% else %}
                #dff0d8;
                {% endif %}; 
                padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p><strong>Severity:</strong> {{ severity }}</p>
        <p><strong>Time:</strong> {{ alert_time }}</p>
        <p><strong>Component:</strong> {{ component }}</p>
        
        <h3>Description:</h3>
        <p>{{ alert_description }}</p>
        
        {% if actions %}
        <h3>Recommended Actions:</h3>
        <ul>
            {% for action in actions %}
            <li>{{ action }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        <p><strong>Alert ID:</strong> {{ alert_id }}</p>
    </div>
    
    <p style="color: #777; font-size: 12px; text-align: center; margin-top: 30px;">AuditPulse System Monitoring</p>
</div>
        """,
        "description": "System alert notification for administrators",
        "placeholders": {
            "alert_type": "Type of alert",
            "alert_title": "Alert title",
            "severity": "Alert severity (critical, high, medium, low)",
            "alert_time": "Time of the alert",
            "component": "System component that triggered the alert",
            "alert_description": "Detailed description of the alert",
            "actions": "List of recommended actions",
            "alert_id": "Unique identifier for the alert",
        },
    },
    # Account security notification
    {
        "template_id": "account_security",
        "name": "Account Security Alert",
        "subject": "Security Alert: {{ alert_type }}",
        "body": """
Dear {{ user_name }},

We're contacting you about a security event related to your AuditPulse account.

Event: {{ alert_type }}
Time: {{ alert_time }}
Location: {{ location }}
IP Address: {{ ip_address }}
Device: {{ device }}

{% if alert_type == 'login_attempt' %}
There were {{ attempt_count }} failed login attempts to your account.
{% endif %}

{% if alert_type == 'password_changed' %}
Your account password was changed. If you didn't make this change, please reset your password immediately.
{% endif %}

{% if alert_type == 'new_device' %}
Your account was accessed from a new device or location.
{% endif %}

If this was you, no action is needed.

If you don't recognize this activity, please:
1. Reset your password immediately
2. Enable two-factor authentication
3. Contact our support team at support@auditpulse.com

Security & Fraud Prevention,
AuditPulse Team
        """,
        "html_body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 5px;">
    <h2 style="color: #d9534f;">Security Alert: {{ alert_type }}</h2>
    <p>Dear {{ user_name }},</p>
    
    <p>We're contacting you about a security event related to your AuditPulse account.</p>
    
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p><strong>Event:</strong> {{ alert_type }}</p>
        <p><strong>Time:</strong> {{ alert_time }}</p>
        <p><strong>Location:</strong> {{ location }}</p>
        <p><strong>IP Address:</strong> {{ ip_address }}</p>
        <p><strong>Device:</strong> {{ device }}</p>
        
        {% if alert_type == 'login_attempt' %}
        <p style="color: #d9534f;"><strong>There were {{ attempt_count }} failed login attempts to your account.</strong></p>
        {% endif %}
        
        {% if alert_type == 'password_changed' %}
        <p style="color: #d9534f;"><strong>Your account password was changed. If you didn't make this change, please reset your password immediately.</strong></p>
        {% endif %}
        
        {% if alert_type == 'new_device' %}
        <p style="color: #d9534f;"><strong>Your account was accessed from a new device or location.</strong></p>
        {% endif %}
    </div>
    
    <p><strong>If this was you, no action is needed.</strong></p>
    
    <div style="background-color: #f2dede; padding: 15px; border-radius: 5px; margin: 20px 0; color: #a94442;">
        <p><strong>If you don't recognize this activity, please:</strong></p>
        <ol>
            <li>Reset your password immediately</li>
            <li>Enable two-factor authentication</li>
            <li>Contact our support team at support@auditpulse.com</li>
        </ol>
    </div>
    
    <p>Security & Fraud Prevention,<br>AuditPulse Team</p>
</div>
        """,
        "description": "Account security notification for users",
        "placeholders": {
            "user_name": "User's full name",
            "alert_type": "Type of security alert",
            "alert_time": "Time of the security event",
            "location": "Geographic location",
            "ip_address": "IP address",
            "device": "Device information",
            "attempt_count": "Number of login attempts",
        },
    },
    # Weekly report notification
    {
        "template_id": "weekly_report",
        "name": "Weekly Financial Report",
        "subject": "Your Weekly Financial Report ({{ week_start }} - {{ week_end }})",
        "body": """
Dear {{ user_name }},

Here's your weekly financial summary for {{ week_start }} to {{ week_end }}.

Account Summary:
{% for account in accounts %}
- {{ account.name }}: {{ account.balance }}
{% endfor %}

Transaction Summary:
- Total deposits: {{ total_deposits }}
- Total withdrawals: {{ total_withdrawals }}
- Net change: {{ net_change }}

{% if anomalies %}
Detected Anomalies:
{% for anomaly in anomalies %}
- {{ anomaly.date }}: {{ anomaly.description }} ({{ anomaly.amount }})
{% endfor %}
{% endif %}

Top Spending Categories:
{% for category in top_categories %}
- {{ category.name }}: {{ category.amount }} ({{ category.percentage }}%)
{% endfor %}

To view your complete financial report, visit:
{{ report_url }}

Best regards,
The AuditPulse Team
        """,
        "html_body": """
<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eee; border-radius: 5px;">
    <h2 style="color: #428bca;">Your Weekly Financial Report</h2>
    <p style="color: #777;">{{ week_start }} - {{ week_end }}</p>
    <p>Dear {{ user_name }},</p>
    
    <p>Here's your weekly financial summary.</p>
    
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0;">Account Summary:</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="text-align: left; padding: 8px;">Account</th>
                    <th style="text-align: right; padding: 8px;">Balance</th>
                </tr>
            </thead>
            <tbody>
                {% for account in accounts %}
                <tr>
                    <td style="border-top: 1px solid #ddd; padding: 8px;">{{ account.name }}</td>
                    <td style="border-top: 1px solid #ddd; padding: 8px; text-align: right;">{{ account.balance }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0;">Transaction Summary:</h3>
        <p><strong>Total deposits:</strong> <span style="color: #5cb85c;">{{ total_deposits }}</span></p>
        <p><strong>Total withdrawals:</strong> <span style="color: #d9534f;">{{ total_withdrawals }}</span></p>
        <p><strong>Net change:</strong> <span style="color: {% if net_change.startswith('-') %}#d9534f{% else %}#5cb85c{% endif %};">{{ net_change }}</span></p>
    </div>
    
    {% if anomalies %}
    <div style="background-color: #fcf8e3; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0; color: #f0ad4e;">Detected Anomalies:</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="text-align: left; padding: 8px;">Date</th>
                    <th style="text-align: left; padding: 8px;">Description</th>
                    <th style="text-align: right; padding: 8px;">Amount</th>
                </tr>
            </thead>
            <tbody>
                {% for anomaly in anomalies %}
                <tr>
                    <td style="border-top: 1px solid #ddd; padding: 8px;">{{ anomaly.date }}</td>
                    <td style="border-top: 1px solid #ddd; padding: 8px;">{{ anomaly.description }}</td>
                    <td style="border-top: 1px solid #ddd; padding: 8px; text-align: right;">{{ anomaly.amount }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
    
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0;">Top Spending Categories:</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr>
                    <th style="text-align: left; padding: 8px;">Category</th>
                    <th style="text-align: right; padding: 8px;">Amount</th>
                    <th style="text-align: right; padding: 8px;">Percentage</th>
                </tr>
            </thead>
            <tbody>
                {% for category in top_categories %}
                <tr>
                    <td style="border-top: 1px solid #ddd; padding: 8px;">{{ category.name }}</td>
                    <td style="border-top: 1px solid #ddd; padding: 8px; text-align: right;">{{ category.amount }}</td>
                    <td style="border-top: 1px solid #ddd; padding: 8px; text-align: right;">{{ category.percentage }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <p><a href="{{ report_url }}" style="display: inline-block; background-color: #428bca; color: white; padding: 10px 15px; text-decoration: none; border-radius: 3px; margin-top: 10px;">View Full Report</a></p>
    
    <p>Best regards,<br>The AuditPulse Team</p>
</div>
        """,
        "description": "Weekly financial report notification",
        "placeholders": {
            "user_name": "User's full name",
            "week_start": "Start date of the week",
            "week_end": "End date of the week",
            "accounts": "List of accounts with names and balances",
            "total_deposits": "Total amount of deposits",
            "total_withdrawals": "Total amount of withdrawals",
            "net_change": "Net change in balance",
            "anomalies": "List of detected anomalies",
            "top_categories": "List of top spending categories",
            "report_url": "URL to the full report",
        },
    },
]


async def create_default_templates(template_manager: TemplateManager):
    """Create default notification templates in the database.

    Args:
        template_manager: Template manager instance
    """
    logger.info("Creating default notification templates")

    for template_data in DEFAULT_TEMPLATES:
        try:
            await template_manager.create_template(
                template_id=template_data["template_id"],
                name=template_data["name"],
                subject=template_data["subject"],
                body=template_data["body"],
                html_body=template_data.get("html_body"),
                description=template_data.get("description"),
                placeholders=template_data.get("placeholders"),
            )
            logger.info(f"Created template: {template_data['template_id']}")
        except Exception as e:
            logger.warning(
                f"Failed to create template {template_data['template_id']}: {e}"
            )

    logger.info("Default templates creation completed")
