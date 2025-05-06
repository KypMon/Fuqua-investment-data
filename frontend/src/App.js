import { Container, Typography } from "@mui/material";
import { useState } from "react";
import FormSection from "./components/FormSection";
import ResultSection from "./components/ResultSection";

function App() {
  const [result, setResult] = useState(null);

  return (
    <Container maxWidth="md" sx={{ paddingY: 4 }}>
      <Typography variant="h4" gutterBottom>
        Portfolio Optimizer
      </Typography>
      <FormSection setResult={setResult} />
      <ResultSection result={result} />
    </Container>
  );
}

export default App;
