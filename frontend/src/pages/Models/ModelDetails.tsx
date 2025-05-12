import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// Mock data for the charts
const performanceData = [
  { date: '2024-03-01', accuracy: 0.85, precision: 0.82, recall: 0.88 },
  { date: '2024-03-08', accuracy: 0.87, precision: 0.84, recall: 0.90 },
  { date: '2024-03-15', accuracy: 0.89, precision: 0.86, recall: 0.92 },
  { date: '2024-03-22', accuracy: 0.92, precision: 0.89, recall: 0.94 },
];

const predictionData = [
  { timestamp: '2024-03-22 10:00', prediction: 0.95, actual: 0.92 },
  { timestamp: '2024-03-22 11:00', prediction: 0.93, actual: 0.91 },
  { timestamp: '2024-03-22 12:00', prediction: 0.94, actual: 0.93 },
  { timestamp: '2024-03-22 13:00', prediction: 0.96, actual: 0.95 },
];

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`model-tabpanel-${index}`}
      aria-labelledby={`model-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const ModelDetails: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Transaction Anomaly Detector
        </Typography>
        <Typography color="textSecondary" gutterBottom>
          Version 1.2.0
        </Typography>
        <Chip label="Active" color="success" sx={{ mt: 1 }} />
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Overview
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Accuracy
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ flexGrow: 1, mr: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={95}
                      color="primary"
                    />
                  </Box>
                  <Typography variant="body2">95%</Typography>
                </Box>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Precision
                </Typography>
                <Typography variant="body1">92%</Typography>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Recall
                </Typography>
                <Typography variant="body1">88%</Typography>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  F1 Score
                </Typography>
                <Typography variant="body1">90%</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Tabs value={tabValue} onChange={handleTabChange}>
                <Tab label="Performance" />
                <Tab label="Predictions" />
                <Tab label="Configuration" />
              </Tabs>

              <TabPanel value={tabValue} index={0}>
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#8884d8"
                        name="Accuracy"
                      />
                      <Line
                        type="monotone"
                        dataKey="precision"
                        stroke="#82ca9d"
                        name="Precision"
                      />
                      <Line
                        type="monotone"
                        dataKey="recall"
                        stroke="#ffc658"
                        name="Recall"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Timestamp</TableCell>
                        <TableCell align="right">Prediction</TableCell>
                        <TableCell align="right">Actual</TableCell>
                        <TableCell align="right">Difference</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {predictionData.map((row) => (
                        <TableRow key={row.timestamp}>
                          <TableCell>{row.timestamp}</TableCell>
                          <TableCell align="right">
                            {(row.prediction * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell align="right">
                            {(row.actual * 100).toFixed(1)}%
                          </TableCell>
                          <TableCell align="right">
                            {(
                              Math.abs(row.prediction - row.actual) * 100
                            ).toFixed(1)}
                            %
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </TabPanel>

              <TabPanel value={tabValue} index={2}>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="subtitle1" gutterBottom>
                      Model Parameters
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableBody>
                          <TableRow>
                            <TableCell>Learning Rate</TableCell>
                            <TableCell>0.001</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Batch Size</TableCell>
                            <TableCell>32</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Epochs</TableCell>
                            <TableCell>100</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>Optimizer</TableCell>
                            <TableCell>Adam</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                </Grid>
              </TabPanel>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}; 