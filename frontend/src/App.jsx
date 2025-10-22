import { useState } from "react";
import { AppBar, Tabs, Tab, Toolbar, Typography, Box, Container } from "@mui/material";
import FormSection from "./components/FormSection";
import ResultSection from "./components/ResultSection";
import BacktestForm from "./components/BacktestForm";
import BacktestResult from "./components/BacktestResult";
import RegressionPage from "./components/RegressionPage";
import MatrixPage from "./components/MatrixPage";
import { useNavigate, useLocation, Routes, Route, Navigate, NavLink  } from "react-router-dom";

function App() {
  const [result, setResult] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);

  const location = useLocation();
  const navigate = useNavigate();

  // Match the current path to tab value
  const currentPath = location.pathname;
  const tabValue = currentPath.startsWith("/matrix")
    ? "matrix"
    : currentPath.startsWith("/backtest")
    ? "backtest"
    : currentPath.startsWith("/regression")
    ? "regression"
    : "mv";
    
  const handleTabChange = (event, newValue) => {
    navigate(`/${newValue}`);
  };

  return (
    <>
      <AppBar position="static" color="primary">
        <Toolbar>
          <Typography
            variant="h6"
            sx={{ flexGrow: 1, cursor: "pointer" }}
            onClick={() => navigate("/mv")}
          >
            Fuqua Finance Analyzer
          </Typography>

          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            indicatorColor="secondary"
            textColor="inherit"
          >
            <Tab label="MV Analysis" value="mv" />
            <Tab label="Backtest" value="backtest" />
            <Tab label="Regression" value="regression" />
            <Tab label="Matrix" value="matrix" />
          </Tabs>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ paddingY: 4 }}>
        <Routes>
          <Route path="/" element={<Navigate to="/mv" replace />} />
          <Route
            path="/mv"
            element={
              <>
                <FormSection setResult={setResult} />
                <ResultSection result={result} />
              </>
            }
          />
          <Route
            path="/matrix"
            element={<MatrixPage />}
          />
          <Route path="/backtest" element={
            <>
              <BacktestForm setBacktestResult={setBacktestResult} />
              <BacktestResult result={backtestResult} />
            </>
          }/>
          <Route
            path="/regression"
            element={
              <>
                <RegressionPage />
              </>
            }
          />
        </Routes>
      </Container>
    </>
  );
}

export default App;
