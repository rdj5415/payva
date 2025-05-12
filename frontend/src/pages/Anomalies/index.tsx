import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Grid,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Visibility as VisibilityIcon,
} from '@mui/icons-material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';

const mockAnomalies = [
  {
    id: 1,
    date: '2024-03-20',
    type: 'Unusual Transaction',
    severity: 'High',
    status: 'Open',
    description: 'Large transaction amount detected',
    amount: 50000.00,
  },
  {
    id: 2,
    date: '2024-03-19',
    type: 'Pattern Change',
    severity: 'Medium',
    status: 'Investigating',
    description: 'Unusual transaction pattern detected',
    amount: 15000.00,
  },
  // Add more mock data as needed
];

const columns: GridColDef[] = [
  { field: 'date', headerName: 'Date', width: 130 },
  { field: 'type', headerName: 'Type', width: 150 },
  {
    field: 'severity',
    headerName: 'Severity',
    width: 130,
    renderCell: (params) => {
      const severity = params.value as string;
      const color =
        severity === 'High'
          ? 'error'
          : severity === 'Medium'
          ? 'warning'
          : 'info';
      return (
        <Chip
          icon={
            severity === 'High' ? (
              <ErrorIcon />
            ) : severity === 'Medium' ? (
              <WarningIcon />
            ) : (
              <InfoIcon />
            )
          }
          label={severity}
          color={color}
          size="small"
        />
      );
    },
  },
  { field: 'status', headerName: 'Status', width: 130 },
  {
    field: 'amount',
    headerName: 'Amount',
    width: 130,
    valueFormatter: (params) => {
      const value = params.value as number;
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
      }).format(value);
    },
  },
  {
    field: 'actions',
    headerName: 'Actions',
    width: 100,
    renderCell: (params) => (
      <IconButton
        size="small"
        onClick={() => handleViewDetails(params.row)}
      >
        <VisibilityIcon />
      </IconButton>
    ),
  },
];

export const Anomalies: React.FC = () => {
  const [selectedAnomaly, setSelectedAnomaly] = useState<any>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const handleViewDetails = (anomaly: any) => {
    setSelectedAnomaly(anomaly);
    setIsDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setIsDialogOpen(false);
    setSelectedAnomaly(null);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Anomalies
      </Typography>

      <Paper sx={{ height: 600 }}>
        <DataGrid
          rows={mockAnomalies}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[10]}
          disableSelectionOnClick
        />
      </Paper>

      <Dialog open={isDialogOpen} onClose={handleCloseDialog} maxWidth="md">
        {selectedAnomaly && (
          <>
            <DialogTitle>Anomaly Details</DialogTitle>
            <DialogContent>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Date</Typography>
                  <Typography>{selectedAnomaly.date}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Type</Typography>
                  <Typography>{selectedAnomaly.type}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Severity</Typography>
                  <Typography>{selectedAnomaly.severity}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="subtitle2">Status</Typography>
                  <Typography>{selectedAnomaly.status}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2">Description</Typography>
                  <Typography>{selectedAnomaly.description}</Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="subtitle2">Amount</Typography>
                  <Typography>
                    {new Intl.NumberFormat('en-US', {
                      style: 'currency',
                      currency: 'USD',
                    }).format(selectedAnomaly.amount)}
                  </Typography>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseDialog}>Close</Button>
              <Button variant="contained" color="primary">
                Take Action
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
}; 