import { useState } from "react";
import { AppBar, Tabs, Tab, Toolbar, Typography, Box, Container } from "@mui/material";
import FormSection from "./components/FormSection";
import ResultSection from "./components/ResultSection";

function App() {
  const [result, setResult] = useState(null);
  const [tabIndex, setTabIndex] = useState(0);

  const handleTabChange = (event, newValue) => {
    setTabIndex(newValue);
  };

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Fuqua Finance Analyzer
          </Typography>
          <Tabs value={tabIndex} onChange={handleTabChange} textColor="inherit" indicatorColor="secondary">
            <Tab label="MV Analysis" />
            <Tab label="Backtest" />
          </Tabs>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ paddingY: 4 }}>
        {tabIndex === 0 && (
          <>
            <FormSection setResult={setResult} />
            <ResultSection result={result} />
          </>
        )}
        {tabIndex === 1 && (
          <Box>
            <Typography variant="h5">🔧 Backtest (coming soon...)</Typography>
            <Typography mt={2}>You can implement backtest logic here.</Typography>
          </Box>
        )}
      </Container>
    </>
  );
}

export default App;
