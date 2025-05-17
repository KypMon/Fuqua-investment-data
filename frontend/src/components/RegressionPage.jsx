import React, { useState } from "react";
import { Box, Typography, CircularProgress } from "@mui/material";
import axios from "axios";
import RegressionForm from "./RegressionForm";
import RegressionResult from "./RegressionResult";

export default function RegressionPage() {
  const [form, setForm] = useState({
    ticker: "AAPL",
    model: "FF3",
    start_date: "2020-01-01",
    end_date: "2023-12-31",
    rolling_period: 36
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/regression`, form);
      setResult(res.data);
    } catch (err) {
      console.error("Regression error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Factor Regression Analysis
      </Typography>

      <RegressionForm form={form} setForm={setForm} onSubmit={handleSubmit} loading={loading} />

      {loading && (
        <Box mt={4} textAlign="center">
          <CircularProgress />
        </Box>
      )}

      <RegressionResult result={result} />
    </Box>
  );
}
