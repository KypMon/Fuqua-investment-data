import React, { useState } from "react";
import { Box, Typography, CircularProgress, Alert } from "@mui/material";
import axios from "axios";
import RegressionForm from "./RegressionForm";
import RegressionResult from "./RegressionResult";
import dayjs from "dayjs";

export default function RegressionPage() {
  const [form, setForm] = useState({
    etflist: ["AAPL"],
    model: "FF3",
    start_date: "2020-01-01",
    end_date: "2023-12-31",
    rolling_period: 36,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState([]);

  const validateForm = () => {
    const validationErrors = [];
    const ticker = form.etflist?.[0]?.trim() || "";

    if (!ticker) {
      validationErrors.push("Please provide a ticker symbol.");
    }

    const start = dayjs(form.start_date);
    const end = dayjs(form.end_date);

    if (!start.isValid()) {
      validationErrors.push("Start date is invalid.");
    }

    if (!end.isValid()) {
      validationErrors.push("End date is invalid.");
    }

    if (start.isValid() && end.isValid() && start.isAfter(end)) {
      validationErrors.push("Start date must be before or equal to the end date.");
    }

    const rollingPeriod = Number(form.rolling_period);
    if (!Number.isInteger(rollingPeriod) || rollingPeriod <= 0) {
      validationErrors.push("Rolling period must be a positive integer.");
    }

    return validationErrors;
  };

  const handleSubmit = async () => {
    const validationErrors = validateForm();
    if (validationErrors.length > 0) {
      setErrors(validationErrors);
      setResult(null);
      return;
    }

    setErrors([]);
    setLoading(true);
    setResult(null);
    try {
      const { etflist, ...rest } = form;
      const payload = { ...rest, ticker: etflist[0] };
      const res = await axios.post(
        `${process.env.REACT_APP_API_BASE_URL}/regression`,
        payload
      );
      setResult(res.data);
      setErrors([]);
    } catch (err) {
      console.error("Regression error:", err);
      const responseErrors = err.response?.data?.errors || [err.response?.data?.error || "Failed to run regression. Please try again."];
      setErrors(responseErrors);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Factor Regression Analysis
      </Typography>

      {errors.map((error, idx) => (
        <Alert severity="error" key={`reg-error-${idx}`} sx={{ mt: idx === 0 ? 2 : 1 }}>
          {error}
        </Alert>
      ))}

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
