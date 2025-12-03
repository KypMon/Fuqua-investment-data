import { AppBar, Container, Tab, Tabs, Toolbar, Typography } from "@mui/material";
//import { useState } from "react";
import { useEffect, useRef, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import BacktestForm from "./components/BacktestForm";
import BacktestResult from "./components/BacktestResult";
import FormSection from "./components/FormSection";
import LifeCycleSimulationPage from "./components/LifeCycleSimulationPage";
import MatrixPage from "./components/MatrixPage";
import RegressionPage from "./components/RegressionPage";
import ResultSection from "./components/ResultSection";

function App() {
  // user authentication
  const fwAuthRef = useRef(null);
  useEffect(() => {
    const authElement = fwAuthRef.current;
    if (authElement) {
      authElement.setAttribute('url', `${process.env.REACT_APP_AUTH_URL}`);
      authElement.setAttribute('validateUrl', `${process.env.REACT_APP_VALIDATE_URL}`);
    }
  }, []);

  const [result, setResult] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);

  const location = useLocation();
  const navigate = useNavigate();

  // Match the current path to tab value
  const currentPath = location.pathname;
  // const tabValue = currentPath.startsWith("/matrix")
  //   ? "matrix"
  //   : currentPath.startsWith("/backtest")
  //     ? "backtest"
  //     : currentPath.startsWith("/regression")
  //       ? "regression"
  //       : currentPath.startsWith("/life-cycle")
  //         ? "life-cycle"
  //         : "mv";

  const tabValue = currentPath.startsWith("/matrix")
    ? "matrix"
    : currentPath.startsWith("/backtest")
      ? "backtest"
      : currentPath.startsWith("/regression")
        ? "regression"
        : currentPath.startsWith("/life-cycle")
          ? "life-cycle"
          : currentPath === "/" || currentPath.startsWith("/mv")
            ? "mv"
            : "";


  const handleTabChange = (event, newValue) => {
    navigate(`/${newValue}`);
  };

  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || "";
  const isLocalhost = apiBaseUrl.includes("localhost");
  //console.log(`${apiBaseUrl}`);

  return (
    <>
      {/* Conditionally render fw-auth only if NOT localhost */}
      {!isLocalhost && <fw-auth auto></fw-auth>}

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
            <Tab label="Life Cycle" value="life-cycle" />
          </Tabs>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ paddingY: 4 }}>
        <Routes>
          {/* <Route path="/" element={<Navigate to="/mv" replace />} /> */}
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
          <Route path="/life-cycle" element={<LifeCycleSimulationPage />} />
          <Route path="/backtest" element={
            <>
              <BacktestForm setBacktestResult={setBacktestResult} />
              <BacktestResult result={backtestResult} />
            </>
          } />
          <Route
            path="/regression"
            element={
              <>
                <RegressionPage />
              </>
            }
          />
          <Route path="*" element={<Navigate to="/mv" replace />} />
        </Routes>
      </Container>
    </>
  );
}

export default App;
