/**
 * AuditPulse Admin Dashboard JavaScript
 * Handles dashboard functionality, API calls, and UI updates
 */

// Global variables for charts
let transactionsChart;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initDashboard();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    loadDashboardData();
});

/**
 * Initialize the dashboard components
 */
function initDashboard() {
    // Initialize tabs
    const triggerTabList = [].slice.call(document.querySelectorAll('#sidebar .nav-link'));
    triggerTabList.forEach(function (triggerEl) {
        const tabTrigger = new bootstrap.Tab(triggerEl);
        
        triggerEl.addEventListener('click', function (event) {
            event.preventDefault();
            tabTrigger.show();
            
            // Update active class
            triggerTabList.forEach(el => el.classList.remove('active'));
            triggerEl.classList.add('active');
            
            // Load data for selected tab
            const tabId = triggerEl.getAttribute('href').substring(1);
            loadTabData(tabId);
        });
    });
    
    // Initialize transactions chart
    const ctx = document.getElementById('transactionsChart');
    if (ctx) {
        transactionsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 30}, (_, i) => i + 1), // Days of month
                datasets: [{
                    label: 'Transactions',
                    data: generateRandomData(30, 50, 200),
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Anomalies',
                    data: generateRandomData(30, 0, 20),
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

/**
 * Set up event listeners for dashboard elements
 */
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            loadDashboardData();
        });
    }
    
    // Logout button
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function(e) {
            e.preventDefault();
            logout();
        });
    }
    
    // Create model button
    const createModelBtn = document.getElementById('create-model-btn');
    if (createModelBtn) {
        createModelBtn.addEventListener('click', function() {
            createNewModel();
        });
    }
}

/**
 * Load dashboard data and update UI
 */
function loadDashboardData() {
    showLoading();
    
    // Simulate API calls with setTimeout
    setTimeout(function() {
        // Update counters
        updateCounters();
        
        // Update system health
        updateSystemHealth();
        
        // Update active tab data
        const activeTab = document.querySelector('#sidebar .nav-link.active');
        if (activeTab) {
            const tabId = activeTab.getAttribute('href').substring(1);
            loadTabData(tabId);
        }
        
        hideLoading();
    }, 500);
}

/**
 * Load data for specific tab
 * @param {string} tabId - The ID of the tab to load data for
 */
function loadTabData(tabId) {
    switch(tabId) {
        case 'dashboard':
            // Already handled in loadDashboardData
            break;
        case 'models':
            loadModelsData();
            break;
        case 'system':
            loadSystemData();
            break;
        case 'transactions':
            loadTransactionsData();
            break;
        case 'anomalies':
            loadAnomaliesData();
            break;
        default:
            console.log(`No data loader for tab ${tabId}`);
    }
}

/**
 * Update dashboard counters with mock data
 */
function updateCounters() {
    // Get counter elements
    const transactionCount = document.getElementById('transaction-count');
    const anomalyCount = document.getElementById('anomaly-count');
    const modelCount = document.getElementById('model-count');
    const tenantCount = document.getElementById('tenant-count');
    
    // Set values with animation
    if (transactionCount) animateCounter(transactionCount, 0, 1248);
    if (anomalyCount) animateCounter(anomalyCount, 0, 37);
    if (modelCount) animateCounter(modelCount, 0, 5);
    if (tenantCount) animateCounter(tenantCount, 0, 12);
}

/**
 * Update system health indicators
 */
function updateSystemHealth() {
    // CPU usage
    const cpuUsage = document.getElementById('cpu-usage');
    if (cpuUsage) {
        const cpuValue = getRandomInt(10, 70);
        cpuUsage.style.width = `${cpuValue}%`;
        cpuUsage.textContent = `${cpuValue}%`;
        cpuUsage.className = `progress-bar ${cpuValue > 80 ? 'bg-danger' : cpuValue > 60 ? 'bg-warning' : 'bg-primary'}`;
    }
    
    // Memory usage
    const memoryUsage = document.getElementById('memory-usage');
    if (memoryUsage) {
        const memoryValue = getRandomInt(20, 60);
        memoryUsage.style.width = `${memoryValue}%`;
        memoryUsage.textContent = `${memoryValue}%`;
        memoryUsage.className = `progress-bar ${memoryValue > 80 ? 'bg-danger' : memoryValue > 60 ? 'bg-warning' : 'bg-success'}`;
    }
    
    // Disk usage
    const diskUsage = document.getElementById('disk-usage');
    if (diskUsage) {
        const diskValue = getRandomInt(30, 70);
        diskUsage.style.width = `${diskValue}%`;
        diskUsage.textContent = `${diskValue}%`;
        diskUsage.className = `progress-bar ${diskValue > 80 ? 'bg-danger' : diskValue > 60 ? 'bg-warning' : 'bg-info'}`;
    }
    
    // Service statuses
    updateServiceStatus('api-status', getRandomInt(1, 10) > 1);
    updateServiceStatus('db-status', getRandomInt(1, 10) > 1);
    updateServiceStatus('queue-status', getRandomInt(1, 10) > 1);
    
    // Overall system status
    const systemStatus = document.getElementById('system-status');
    if (systemStatus) {
        const allHealthy = document.querySelectorAll('.badge.bg-success').length === 3;
        systemStatus.textContent = allHealthy ? 'System Operational' : 'System Degraded';
        systemStatus.className = `badge ${allHealthy ? 'bg-success' : 'bg-warning'}`;
    }
}

/**
 * Update a service status badge
 * @param {string} id - The ID of the status element
 * @param {boolean} isHealthy - Whether the service is healthy
 */
function updateServiceStatus(id, isHealthy) {
    const statusElement = document.getElementById(id);
    if (statusElement) {
        statusElement.textContent = isHealthy ? 'Healthy' : 'Degraded';
        statusElement.className = `badge ${isHealthy ? 'bg-success' : 'bg-warning'} rounded-pill`;
    }
}

/**
 * Load models data for the models tab
 */
function loadModelsData() {
    // This would be an API call in a real application
    console.log('Loading models data...');
    
    // For demo purposes, we'll just update the system-health-details div
    const systemHealthDetails = document.getElementById('system-health-details');
    if (systemHealthDetails) {
        systemHealthDetails.innerHTML = `
            <div class="alert alert-info">
                <h6>System Information</h6>
                <p><strong>Server:</strong> AuditPulse-Production-01</p>
                <p><strong>Version:</strong> 1.2.3</p>
                <p><strong>Uptime:</strong> 15 days, 7 hours</p>
                <p><strong>Last Restart:</strong> 2023-10-01 00:00:00</p>
            </div>
        `;
    }
}

/**
 * Load system data for the system tab
 */
function loadSystemData() {
    // Populate error logs table with mock data
    const errorLogsTable = document.getElementById('error-logs-table');
    if (errorLogsTable) {
        let html = '';
        
        // Generate mock error logs
        for (let i = 0; i < 5; i++) {
            const date = new Date();
            date.setHours(date.getHours() - i);
            
            html += `
                <tr>
                    <td>${date.toISOString().replace('T', ' ').substring(0, 19)}</td>
                    <td>${['API', 'Database', 'Worker', 'Authentication'][getRandomInt(0, 3)]}</td>
                    <td>${['ERROR', 'WARNING', 'INFO'][getRandomInt(0, 2)]}</td>
                    <td>${getRandomErrorMessage()}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary">Details</button>
                        <button class="btn btn-sm btn-outline-danger">Dismiss</button>
                    </td>
                </tr>
            `;
        }
        
        errorLogsTable.innerHTML = html;
    }
}

/**
 * Create a new model (mock implementation)
 */
function createNewModel() {
    // Get form values
    const modelType = document.getElementById('model-type').value;
    const modelFile = document.getElementById('model-file').files[0];
    const description = document.getElementById('model-description').value;
    const activateImmediately = document.getElementById('activate-model').checked;
    
    // Validate form
    if (!modelType || !modelFile) {
        alert('Please fill in all required fields');
        return;
    }
    
    // Show loading
    showLoading();
    
    // Simulate API call
    setTimeout(function() {
        // Close modal
        const modalElement = document.getElementById('newModelModal');
        const modal = bootstrap.Modal.getInstance(modalElement);
        modal.hide();
        
        // Show success message
        alert('Model created successfully!');
        
        // Refresh models data
        loadModelsData();
        
        // Hide loading
        hideLoading();
    }, 1500);
}

/**
 * Handle logout (mock implementation)
 */
function logout() {
    if (confirm('Are you sure you want to log out?')) {
        // Redirect to login page
        window.location.href = '/login';
    }
}

/**
 * Animate counting up for dashboard counters
 * @param {Element} element - The element to update
 * @param {number} start - Starting count value
 * @param {number} end - Ending count value
 * @param {number} duration - Animation duration in milliseconds
 */
function animateCounter(element, start, end, duration = 1000) {
    const range = end - start;
    const step = Math.ceil(range / 100);
    const increment = Math.ceil(range / (duration / 10));
    
    let current = start;
    const timer = setInterval(function() {
        current += increment;
        if (current >= end) {
            element.textContent = end.toLocaleString();
            clearInterval(timer);
        } else {
            element.textContent = current.toLocaleString();
        }
    }, 10);
}

/**
 * Show loading indicator
 */
function showLoading() {
    // Add loading class to body
    document.body.classList.add('loading');
    
    // Disable buttons
    const buttons = document.querySelectorAll('button:not([data-bs-dismiss])');
    buttons.forEach(button => {
        button.disabled = true;
    });
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    // Remove loading class from body
    document.body.classList.remove('loading');
    
    // Enable buttons
    const buttons = document.querySelectorAll('button:not([data-bs-dismiss])');
    buttons.forEach(button => {
        button.disabled = false;
    });
}

/**
 * Generate random data for charts
 * @param {number} count - Number of data points
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {Array} Array of random values
 */
function generateRandomData(count, min, max) {
    return Array.from({length: count}, () => getRandomInt(min, max));
}

/**
 * Get random integer between min and max
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {number} Random integer
 */
function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Get random error message for mock data
 * @returns {string} Random error message
 */
function getRandomErrorMessage() {
    const messages = [
        'Database connection timed out',
        'Failed to authenticate user',
        'API rate limit exceeded',
        'Unable to process transaction',
        'Invalid configuration detected',
        'Memory allocation error',
        'Failed to load model file',
        'Network connection interrupted'
    ];
    
    return messages[getRandomInt(0, messages.length - 1)];
}

// Function stubs for other data loaders
function loadTransactionsData() {
    console.log('Loading transactions data...');
}

function loadAnomaliesData() {
    console.log('Loading anomalies data...');
} 