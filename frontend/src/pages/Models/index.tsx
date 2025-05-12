import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
} from '@mui/icons-material';

const mockModels = [
  {
    id: 1,
    name: 'Transaction Anomaly Detector',
    version: '1.2.0',
    status: 'Active',
    accuracy: 0.95,
    lastUpdated: '2024-03-20',
    performance: {
      precision: 0.92,
      recall: 0.88,
      f1Score: 0.90,
    },
  },
  {
    id: 2,
    name: 'Pattern Recognition Model',
    version: '2.1.0',
    status: 'Training',
    accuracy: 0.89,
    lastUpdated: '2024-03-19',
    performance: {
      precision: 0.87,
      recall: 0.85,
      f1Score: 0.86,
    },
  },
  // Add more mock data as needed
];

export const Models: React.FC = () => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState<any>(null);

  const handleDeployModel = (model: any) => {
    setSelectedModel(model);
    setIsDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setIsDialogOpen(false);
    setSelectedModel(null);
  };

  const handleConfirmDeploy = () => {
    // Implement model deployment logic
    console.log('Deploying model:', selectedModel);
    handleCloseDialog();
  };

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Typography variant="h4">Models</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => handleDeployModel({})}
        >
          Deploy New Model
        </Button>
      </Box>

      <Grid container spacing={3}>
        {mockModels.map((model) => (
          <Grid item xs={12} md={6} key={model.id}>
            <Card>
              <CardContent>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    mb: 2,
                  }}
                >
                  <Box>
                    <Typography variant="h6">{model.name}</Typography>
                    <Typography color="textSecondary">
                      Version {model.version}
                    </Typography>
                  </Box>
                  <Chip
                    label={model.status}
                    color={model.status === 'Active' ? 'success' : 'warning'}
                    size="small"
                  />
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Accuracy
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box sx={{ flexGrow: 1, mr: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={model.accuracy * 100}
                        color="primary"
                      />
                    </Box>
                    <Typography variant="body2">
                      {(model.accuracy * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">
                      Precision
                    </Typography>
                    <Typography variant="body1">
                      {(model.performance.precision * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">
                      Recall
                    </Typography>
                    <Typography variant="body1">
                      {(model.performance.recall * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={4}>
                    <Typography variant="body2" color="textSecondary">
                      F1 Score
                    </Typography>
                    <Typography variant="body1">
                      {(model.performance.f1Score * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                </Grid>

                <Typography
                  variant="body2"
                  color="textSecondary"
                  sx={{ mt: 2 }}
                >
                  Last Updated: {model.lastUpdated}
                </Typography>
              </CardContent>

              <CardActions>
                <Button
                  size="small"
                  startIcon={<PlayIcon />}
                  disabled={model.status === 'Active'}
                >
                  Activate
                </Button>
                <Button
                  size="small"
                  startIcon={<StopIcon />}
                  disabled={model.status !== 'Active'}
                >
                  Deactivate
                </Button>
                <Button size="small" startIcon={<RefreshIcon />}>
                  Retrain
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={isDialogOpen} onClose={handleCloseDialog} maxWidth="sm">
        <DialogTitle>
          {selectedModel?.id ? 'Update Model' : 'Deploy New Model'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <TextField
              fullWidth
              label="Model Name"
              margin="normal"
              defaultValue={selectedModel?.name}
            />
            <TextField
              fullWidth
              label="Version"
              margin="normal"
              defaultValue={selectedModel?.version}
            />
            <TextField
              fullWidth
              label="Description"
              margin="normal"
              multiline
              rows={4}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button variant="contained" onClick={handleConfirmDeploy}>
            {selectedModel?.id ? 'Update' : 'Deploy'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}; 