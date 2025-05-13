from setuptools import setup, find_packages

setup(
    name="auditpulse_mvp",
    version="0.1.0",
    description="AuditPulse AI MVP - Financial Transaction Anomaly Detection",
    author="AuditPulse Team",
    author_email="info@auditpulse.ai",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core requirements will be read from requirements.txt
    ],
    entry_points={
        "console_scripts": [
            "auditpulse=auditpulse_mvp.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Accounting",
    ],
)
