import React from 'react';
import { Link } from 'react-router-dom';
import { Box, Typography, Button, Container } from '@mui/material';

const NotFound: React.FC = () => {
  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
        }}
      >
        <Typography
          variant="h1"
          component="h1"
          sx={{
            fontSize: '9rem',
            fontWeight: 'bold',
            color: 'primary.main',
          }}
        >
          404
        </Typography>
        <Typography
          variant="h2"
          component="h2"
          sx={{
            mt: 2,
            fontWeight: 'medium',
          }}
        >
          Page Not Found
        </Typography>
        <Typography
          variant="body1"
          color="text.secondary"
          sx={{ mt: 1 }}
        >
          The page you're looking for doesn't exist or has been moved.
        </Typography>
        <Button
          component={Link}
          to="/"
          variant="contained"
          size="large"
          sx={{ mt: 4 }}
        >
          Return Home
        </Button>
      </Box>
    </Container>
  );
};

export default NotFound; 