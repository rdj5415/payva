import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Grid,
  MenuItem,
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { DataGrid, GridColDef } from '@mui/x-data-grid';

const mockTransactions = [
  {
    id: 1,
    date: '2024-03-20',
    description: 'Office Supplies',
    amount: -125.50,
    category: 'Expenses',
    status: 'Processed',
  },
  {
    id: 2,
    date: '2024-03-19',
    description: 'Client Payment',
    amount: 2500.00,
    category: 'Income',
    status: 'Pending',
  },
  // Add more mock data as needed
];

const columns: GridColDef[] = [
  { field: 'date', headerName: 'Date', width: 130 },
  { field: 'description', headerName: 'Description', width: 200 },
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
  { field: 'category', headerName: 'Category', width: 130 },
  {
    field: 'status',
    headerName: 'Status',
    width: 130,
    renderCell: (params) => (
      <Box
        sx={{
          backgroundColor:
            params.value === 'Processed' ? 'success.light' : 'warning.light',
          color: 'white',
          padding: '4px 8px',
          borderRadius: 1,
        }}
      >
        {params.value}
      </Box>
    ),
  },
];

export const Transactions: React.FC = () => {
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);
  const [category, setCategory] = useState('');

  const handleFilter = () => {
    // Implement filter logic
    console.log('Filtering with:', { startDate, endDate, category });
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Transactions
      </Typography>

      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <DatePicker
              label="Start Date"
              value={startDate}
              onChange={(newValue) => setStartDate(newValue)}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <DatePicker
              label="End Date"
              value={endDate}
              onChange={(newValue) => setEndDate(newValue)}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField
              select
              fullWidth
              label="Category"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="Income">Income</MenuItem>
              <MenuItem value="Expenses">Expenses</MenuItem>
            </TextField>
          </Grid>
          <Grid item xs={12} md={3}>
            <Button
              variant="contained"
              fullWidth
              onClick={handleFilter}
              sx={{ height: '56px' }}
            >
              Apply Filters
            </Button>
          </Grid>
        </Grid>
      </Paper>

      <Paper sx={{ height: 600 }}>
        <DataGrid
          rows={mockTransactions}
          columns={columns}
          pageSize={10}
          rowsPerPageOptions={[10]}
          checkboxSelection
          disableSelectionOnClick
        />
      </Paper>
    </Box>
  );
}; 