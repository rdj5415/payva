import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';

const mockData = {
  transactions: {
    total: 1234,
    pending: 56,
    processed: 1178,
  },
  anomalies: {
    total: 89,
    critical: 12,
    warning: 77,
  },
  models: {
    active: 3,
    accuracy: 0.95,
    lastUpdated: '2024-03-20',
  },
};

const chartData = [
  { name: 'Jan', transactions: 4000, anomalies: 2400 },
  { name: 'Feb', transactions: 3000, anomalies: 1398 },
  { name: 'Mar', transactions: 2000, anomalies: 9800 },
  { name: 'Apr', transactions: 2780, anomalies: 3908 },
  { name: 'May', transactions: 1890, anomalies: 4800 },
  { name: 'Jun', transactions: 2390, anomalies: 3800 },
];

export const Dashboard: React.FC = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Transactions" />
            <CardContent>
              <Typography variant="h3">{mockData.transactions.total}</Typography>
              <Typography color="textSecondary">
                {mockData.transactions.pending} pending
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Anomalies" />
            <CardContent>
              <Typography variant="h3">{mockData.anomalies.total}</Typography>
              <Typography color="error">
                {mockData.anomalies.critical} critical
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader title="Models" />
            <CardContent>
              <Typography variant="h3">{mockData.models.active}</Typography>
              <Typography color="textSecondary">
                {mockData.models.accuracy * 100}% accuracy
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Transactions vs Anomalies
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="transactions" fill="#8884d8" />
                <Bar dataKey="anomalies" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Anomaly Trends
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="anomalies"
                  stroke="#8884d8"
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}; 