import { AppBar, Container, Tab, Tabs, Toolbar, Typography } from "@mui/material";
import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import BacktestForm from "./components/BacktestForm";
import BacktestResult from "./components/BacktestResult";
import FormSection from "./components/FormSection";
import FwAuthWrapper from "./components/FwAuthWrapper";
import LifeCycleSimulationPage from "./components/LifeCycleSimulationPage";
import MatrixPage from "./components/MatrixPage";
import RegressionPage from "./components/RegressionPage";
import ResultSection from "./components/ResultSection";

function App() {

  // -----------------------------------
  // 1. Handle token from <fw-auth>
  // -----------------------------------
  const handleAuthData = async (jwt) => {
    try {
      const res = await fetch(`${process.env.REACT_APP_VALIDATE_JWT_URL}/auth/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jwt }),
        credentials: "include",   // allow backend to set cookie
      });
      if (res.ok) {
        console.log("✅ Session cookie set on backend");
      } else {
        console.error("❌ backend refused jwt", await res.text());
      }
    } catch (err) {
      console.error("❌ error sending jwt to backend:", err);
    }
  };

  const [result, setResult] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);

  const location = useLocation();
  const navigate = useNavigate();

  // -----------------------------------
  // 2. Check existing session on initial load
  // -----------------------------------
  useEffect(() => {
    fetch(`${process.env.REACT_APP_VALIDATE_JWT_URL}/auth/check`, { credentials: "include" })
      .then((res) => {
        console.log("res", res);
        if (res.redirected) {
          console.log("REDIRECTED: res", res);
          // backend issued a 302 redirect to login
          window.location = res.url;
        }
        else if (!res.ok) {
          // 401, 403, etc
          console.warn("user not authenticated; go to login");
          //window.location = "/login";
          //REACT_APP_FW_LOGIN_URL=https://go-dev.fuqua.duke.edu/auth/onelink?service=
          //REACT_APP_HOME_PAGE_REDIRECT=http://localhost.fuqua.duke.edu:5001/mv
          //console.log(`${process.env.REACT_APP_FW_LOGIN_URL}${process.env.REACT_APP_HOME_PAGE_REDIRECT}`);
          //window.location = `${process.env.REACT_APP_FW_LOGIN_URL}${process.env.REACT_APP_HOME_PAGE_REDIRECT}`;
        }
        else {
          console.log("session valid");
        }
      })
      .catch((err) => {
        console.error("auth check failed:", err);
      });
  }, []);


  // Match the current path to tab value
  const currentPath = location.pathname;
  const tabValue = currentPath.startsWith("/matrix")
    ? "matrix"
    : currentPath.startsWith("/backtest")
      ? "backtest"
      : currentPath.startsWith("/regression")
        ? "regression"
        : currentPath.startsWith("/life-cycle")
          ? "life-cycle"
          : "mv";

  const handleTabChange = (event, newValue) => {
    navigate(`/${newValue}`);
  };

  return (
    <>
      <FwAuthWrapper
        validateUrl={process.env.REACT_APP_VALIDATE_JWT_URL}
        onAuth={handleAuthData}
      />
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
        </Routes>
      </Container>
    </>
  );
}

export default App;
