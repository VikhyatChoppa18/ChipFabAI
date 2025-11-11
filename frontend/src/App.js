/**
 * ChipFabAI Frontend Dashboard
 * React application for semiconductor manufacturing process optimization
 */
import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  IconButton,
} from '@mui/material';
import {
  TrendingUp,
  Memory,
  Speed,
  ShowChart,
  PlayArrow,
  Refresh,
  CheckCircle,
  Warning,
  Info,
} from '@mui/icons-material';
import axios from 'axios';
import anime from 'animejs/lib/anime.es.js';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#667eea',
    },
    secondary: {
      main: '#764ba2',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1f3a',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 700,
    },
  },
});

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8081';

function App() {
  const [parameters, setParameters] = useState({
    temperature: 200.0,
    pressure: 1.5,
    etch_time: 60.0,
    gas_flow: 100.0,
    chamber_pressure: 5.0,
    wafer_size: 300,
    process_type: 'etching',
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [animating, setAnimating] = useState(false);
  
  const titleRef = useRef(null);
  const cardRefs = useRef([]);
  const chartRef = useRef(null);

  // Animation effects using anime.js library for smooth UI transitions
  useEffect(() => {
    // Animating title on page load with fade-in and slide-up effect
    if (titleRef.current) {
      anime({
        targets: titleRef.current,
        opacity: [0, 1],
        translateY: [-50, 0],
        duration: 1000,
        easing: 'easeOutExpo',
      });
    }

    // Animating cards sequentially on component mount
    // Each card appears with a staggered delay for visual appeal
    cardRefs.current.forEach((card, index) => {
      if (card) {
        anime({
          targets: card,
          opacity: [0, 1],
          translateX: [-100, 0],
          delay: index * 200,
          duration: 800,
          easing: 'easeOutExpo',
        });
      }
    });
  }, []);

  useEffect(() => {
    // Animate prediction results
    if (prediction) {
      setAnimating(true);
      anime({
        targets: '.prediction-card',
        scale: [0.8, 1],
        opacity: [0, 1],
        duration: 600,
        easing: 'easeOutElastic(1, .6)',
        complete: () => setAnimating(false),
      });

      // Animate yield progress
      anime({
        targets: '.yield-progress',
        width: [`0%`, `${prediction.predicted_yield}%`],
        duration: 1500,
        easing: 'easeOutExpo',
        delay: 300,
      });
    }
  }, [prediction]);

  const handleInputChange = (field) => (event) => {
    const value = parseFloat(event.target.value) || 0;
    setParameters((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  // Main prediction handler - calls the API and updates UI
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      // Call the API gateway to get predictions
      const response = await axios.post(`${API_URL}/api/v1/predict`, parameters);
      setPrediction(response.data);
      
      // Store in history for chart visualization
      setHistory((prev) => [
        { ...response.data, timestamp: new Date(), parameters },
        ...prev.slice(0, 9),
      ]);

      // Animate success
      anime({
        targets: '.success-icon',
        scale: [0, 1.2, 1],
        rotate: [0, 360],
        duration: 800,
        easing: 'easeOutElastic(1, .6)',
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction');
      
      // Animate error
      anime({
        targets: '.error-alert',
        translateX: [-100, 0],
        duration: 500,
        easing: 'easeOutExpo',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/api/v1/optimize`, parameters);
      setPrediction(response.data);
      
      setHistory((prev) => [
        { ...response.data, timestamp: new Date(), parameters },
        ...prev.slice(0, 9),
      ]);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to optimize');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = () => {
    setParameters({
      temperature: 200.0,
      pressure: 1.5,
      etch_time: 60.0,
      gas_flow: 100.0,
      chamber_pressure: 5.0,
      wafer_size: 300,
      process_type: 'etching',
    });

    // Animate sample load
    anime({
      targets: '.parameter-input',
      scale: [1, 1.05, 1],
      duration: 400,
      easing: 'easeOutExpo',
    });
  };

  const chartData = history.map((item, index) => ({
    name: `Run ${index + 1}`,
    yield: item.predicted_yield,
    confidence: item.confidence * 100,
  }));

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%)' }}>
        <AppBar position="static" sx={{ background: 'rgba(26, 31, 58, 0.8)', backdropFilter: 'blur(10px)' }}>
          <Toolbar>
            <Memory sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              ChipFabAI
            </Typography>
            <Chip label="GPU Powered" color="primary" icon={<Speed />} />
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ py: 4 }}>
          <Box ref={titleRef} sx={{ textAlign: 'center', mb: 4 }}>
            <Typography
              variant="h3"
              component="h1"
              gutterBottom
              sx={{
                background: 'linear-gradient(45deg, #667eea 30%, #764ba2 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                fontWeight: 700,
              }}
            >
              Semiconductor Manufacturing Process Optimization
            </Typography>
            <Typography variant="h6" color="text.secondary" sx={{ mt: 2 }}>
              AI-Powered Yield Prediction & Process Optimization with NVIDIA L4 GPU
            </Typography>
          </Box>

          <Grid container spacing={3}>
            {/* Input Parameters */}
            <Grid item xs={12} md={6}>
              <Paper
                ref={(el) => (cardRefs.current[0] = el)}
                sx={{ p: 3, background: 'rgba(26, 31, 58, 0.8)', backdropFilter: 'blur(10px)' }}
              >
                <Typography variant="h5" gutterBottom sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
                  <ShowChart sx={{ mr: 1 }} />
                  Process Parameters
                </Typography>

                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      className="parameter-input"
                      fullWidth
                      label="Temperature (°C)"
                      type="number"
                      value={parameters.temperature}
                      onChange={handleInputChange('temperature')}
                      variant="outlined"
                      sx={{ mb: 2 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      className="parameter-input"
                      fullWidth
                      label="Pressure (Torr)"
                      type="number"
                      value={parameters.pressure}
                      onChange={handleInputChange('pressure')}
                      variant="outlined"
                      sx={{ mb: 2 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      className="parameter-input"
                      fullWidth
                      label="Etch Time (s)"
                      type="number"
                      value={parameters.etch_time}
                      onChange={handleInputChange('etch_time')}
                      variant="outlined"
                      sx={{ mb: 2 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      className="parameter-input"
                      fullWidth
                      label="Gas Flow (sccm)"
                      type="number"
                      value={parameters.gas_flow}
                      onChange={handleInputChange('gas_flow')}
                      variant="outlined"
                      sx={{ mb: 2 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      className="parameter-input"
                      fullWidth
                      label="Chamber Pressure (mTorr)"
                      type="number"
                      value={parameters.chamber_pressure}
                      onChange={handleInputChange('chamber_pressure')}
                      variant="outlined"
                      sx={{ mb: 2 }}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      className="parameter-input"
                      fullWidth
                      label="Wafer Size (mm)"
                      type="number"
                      value={parameters.wafer_size}
                      onChange={handleInputChange('wafer_size')}
                      variant="outlined"
                      sx={{ mb: 2 }}
                    />
                  </Grid>
                </Grid>

                <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    onClick={handlePredict}
                    disabled={loading}
                    startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
                    fullWidth
                    sx={{ py: 1.5 }}
                  >
                    Predict Yield
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={handleOptimize}
                    disabled={loading}
                    startIcon={<TrendingUp />}
                    fullWidth
                    sx={{ py: 1.5 }}
                  >
                    Optimize
                  </Button>
                </Box>

                <Button
                  variant="text"
                  onClick={handleLoadSample}
                  startIcon={<Refresh />}
                  fullWidth
                  sx={{ mt: 1 }}
                >
                  Load Sample Data
                </Button>

                {error && (
                  <Alert severity="error" sx={{ mt: 2 }} className="error-alert">
                    {error}
                  </Alert>
                )}
              </Paper>
            </Grid>

            {/* Prediction Results */}
            <Grid item xs={12} md={6}>
              <Paper
                ref={(el) => (cardRefs.current[1] = el)}
                sx={{ p: 3, background: 'rgba(26, 31, 58, 0.8)', backdropFilter: 'blur(10px)' }}
                className="prediction-card"
              >
                <Typography variant="h5" gutterBottom sx={{ mb: 3, display: 'flex', alignItems: 'center' }}>
                  <CheckCircle sx={{ mr: 1, color: 'success.main' }} />
                  Prediction Results
                </Typography>

                {loading && (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <CircularProgress size={60} />
                    <Typography variant="body1" sx={{ mt: 2 }}>
                      Processing with GPU...
                    </Typography>
                  </Box>
                )}

                {prediction && !loading && (
                  <Box>
                    <Box sx={{ mb: 3 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="h6">Predicted Yield</Typography>
                        <Typography variant="h4" sx={{ fontWeight: 700, color: 'primary.main' }}>
                          {prediction.predicted_yield.toFixed(2)}%
                        </Typography>
                      </Box>
                      <Box sx={{ position: 'relative' }}>
                        <LinearProgress
                          variant="determinate"
                          value={prediction.predicted_yield}
                          sx={{
                            height: 20,
                            borderRadius: 1,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            '& .MuiLinearProgress-bar': {
                              borderRadius: 1,
                              background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                            },
                          }}
                          className="yield-progress"
                        />
                      </Box>
                    </Box>

                    <Grid container spacing={2} sx={{ mb: 3 }}>
                      <Grid item xs={6}>
                        <Card sx={{ background: 'rgba(102, 126, 234, 0.2)' }}>
                          <CardContent>
                            <Typography variant="body2" color="text.secondary">
                              Confidence
                            </Typography>
                            <Typography variant="h6">
                              {(prediction.confidence * 100).toFixed(1)}%
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                      <Grid item xs={6}>
                        <Card sx={{ background: 'rgba(118, 75, 162, 0.2)' }}>
                          <CardContent>
                            <Typography variant="body2" color="text.secondary">
                              Processing Time
                            </Typography>
                            <Typography variant="h6">
                              {prediction.processing_time_ms.toFixed(0)}ms
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    </Grid>

                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Optimal Temperature: {prediction.optimal_temperature?.optimal?.toFixed(1)}°C
                        (Range: {prediction.optimal_temperature?.min?.toFixed(1)} - {prediction.optimal_temperature?.max?.toFixed(1)}°C)
                      </Typography>
                      <Typography variant="subtitle1" gutterBottom>
                        Optimal Pressure: {prediction.optimal_pressure?.optimal?.toFixed(2)} Torr
                        (Range: {prediction.optimal_pressure?.min?.toFixed(2)} - {prediction.optimal_pressure?.max?.toFixed(2)} Torr)
                      </Typography>
                    </Box>

                    {prediction.risk_factors && prediction.risk_factors.length > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                          <Warning sx={{ mr: 1, color: 'warning.main' }} />
                          Risk Factors
                        </Typography>
                        {prediction.risk_factors.map((risk, index) => (
                          <Chip
                            key={index}
                            label={risk}
                            color="warning"
                            size="small"
                            sx={{ mr: 1, mb: 1 }}
                          />
                        ))}
                      </Box>
                    )}

                    {prediction.recommendations && prediction.recommendations.length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                          <Info sx={{ mr: 1, color: 'info.main' }} />
                          Recommendations
                        </Typography>
                        <ul style={{ marginLeft: '20px' }}>
                          {prediction.recommendations.map((rec, index) => (
                            <li key={index}>
                              <Typography variant="body2">{rec}</Typography>
                            </li>
                          ))}
                        </ul>
                      </Box>
                    )}

                    {prediction.optimization_score && (
                      <Box sx={{ mt: 2, p: 2, background: 'rgba(102, 126, 234, 0.1)', borderRadius: 1 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Optimization Score
                        </Typography>
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                          {prediction.optimization_score.toFixed(1)}/100
                        </Typography>
                      </Box>
                    )}
                  </Box>
                )}

                {!prediction && !loading && (
                  <Box sx={{ textAlign: 'center', py: 4, opacity: 0.5 }}>
                    <Info sx={{ fontSize: 48, mb: 2 }} />
                    <Typography variant="body1">
                      Enter parameters and click "Predict Yield" to get started
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>

            {/* History Chart */}
            {history.length > 0 && (
              <Grid item xs={12}>
                <Paper
                  ref={chartRef}
                  sx={{ p: 3, background: 'rgba(26, 31, 58, 0.8)', backdropFilter: 'blur(10px)' }}
                >
                  <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                    Prediction History
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                      <XAxis dataKey="name" stroke="rgba(255, 255, 255, 0.7)" />
                      <YAxis stroke="rgba(255, 255, 255, 0.7)" />
                      <Tooltip
                        contentStyle={{
                          background: 'rgba(26, 31, 58, 0.95)',
                          border: '1px solid rgba(102, 126, 234, 0.5)',
                          borderRadius: '8px',
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="yield"
                        stroke="#667eea"
                        strokeWidth={3}
                        dot={{ fill: '#667eea', r: 5 }}
                        name="Yield (%)"
                      />
                      <Line
                        type="monotone"
                        dataKey="confidence"
                        stroke="#764ba2"
                        strokeWidth={3}
                        dot={{ fill: '#764ba2', r: 5 }}
                        name="Confidence (%)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            )}
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;

