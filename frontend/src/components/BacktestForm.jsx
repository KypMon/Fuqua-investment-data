// src/components/BacktestForm.jsx
import React, { useState } from "react";
import { Button, Stack, TextField, Grid, MenuItem, Typography, Alert } from "@mui/material";
import EtfListInput from "./EtfListInput";
import AddIcon from "@mui/icons-material/Add";
import dayjs from "dayjs";
import axios from "axios";

const benchmarkOptions = ["CRSPVW", "None"];

export default function BacktestForm({ setBacktestResult }) {
  const [form, setForm] = useState({
    etflist: ["VOO", "VXUS", "AVUV"],
    allocation1: ["40", "30", "30"],
    allocation2: ["30", "40", "30"],
    allocation3: ["20", "30", "50"],
    benchmark: "CRSPVW",
    rebalancing: "monthly",
    startdate: "2020-01-01",
    enddate: "2023-12-31",
    start_balance: "10000",
  });

  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState([]);

  const validateForm = () => {
    const validationErrors = [];
    const tickers = form.etflist.map((ticker) => ticker?.trim() || "");

    const start = dayjs(form.startdate);
    const end = dayjs(form.enddate);

    if (!start.isValid()) {
      validationErrors.push("Start date is invalid.");
    }

    if (!end.isValid()) {
      validationErrors.push("End date is invalid.");
    }

    if (start.isValid() && end.isValid() && start.isAfter(end)) {
      validationErrors.push("Start date must be before or equal to the end date.");
    }

    const startBalance = Number(form.start_balance);
    if (Number.isNaN(startBalance) || startBalance <= 0) {
      validationErrors.push("Starting balance must be a positive number.");
    }

    if (tickers.filter((t) => t !== "").length === 0) {
      validationErrors.push("Please provide at least one ticker symbol.");
    }

    const allocationSets = [form.allocation1, form.allocation2, form.allocation3];

    allocationSets.forEach((allocation, index) => {
      if (!Array.isArray(allocation)) {
        return;
      }

      const weights = [];
      allocation.forEach((value, tickerIndex) => {
        const rawValue = value?.toString().trim();
        if (!rawValue) {
          return;
        }

        const numericValue = Number(rawValue);
        if (Number.isNaN(numericValue)) {
          validationErrors.push(`Allocation ${index + 1} for ${tickers[tickerIndex] || `position ${tickerIndex + 1}`} must be a number.`);
          return;
        }

        if (numericValue < 0 || numericValue > 100) {
          validationErrors.push(`Allocation ${index + 1} for ${tickers[tickerIndex] || `position ${tickerIndex + 1}`} must be between 0 and 100.`);
        }

        if (numericValue > 0 && !tickers[tickerIndex]) {
          validationErrors.push(`Positive weight in allocation ${index + 1} is missing a ticker symbol at position ${tickerIndex + 1}.`);
        }

        weights.push(numericValue);
      });

      const activeWeights = weights.filter((value) => !Number.isNaN(value));
      if (activeWeights.length > 0) {
        const sum = activeWeights.reduce((acc, value) => acc + value, 0);
        if (Math.abs(sum - 100) > 0.01) {
          validationErrors.push(`Weights in allocation ${index + 1} must add up to 100.`);
        }
      }
    });

    return validationErrors;
  };

  const handleSubmit = async () => {
    const validationErrors = validateForm();
    if (validationErrors.length > 0) {
      setErrors(validationErrors);
      setBacktestResult(null);
      return;
    }

    setErrors([]);
    setLoading(true);
    try {
      const payload = {
        tickers: form.etflist,
        allocation1: form.allocation1,
        allocation2: form.allocation2,
        allocation3: form.allocation3,
        benchmark: [form.benchmark],
        rebalance: form.rebalancing,
        start_date: form.startdate,
        end_date: form.enddate,
        start_balance: form.start_balance,
      };

      const res = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/backtest`, payload);
      setBacktestResult(res.data);
      setErrors([]);
    } catch (err) {
      console.error("Backtest error:", err);
      const responseErrors = err.response?.data?.errors || [err.response?.data?.error || "Failed to run backtest. Please try again."];
      setErrors(responseErrors);
      setBacktestResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Stack spacing={2}>
      {errors.map((error, idx) => (
        <Alert severity="error" key={`error-${idx}`}>
          {error}
        </Alert>
      ))}

      <Typography variant="subtitle1" gutterBottom>
        ETF List
      </Typography>
      <Grid container spacing={0.5}>
          <EtfListInput
            etflist={form.etflist}
            setEtflist={(list) => setForm({ ...form, etflist: list })}
            size={2.3}
          />
      </Grid>


      <Button
        variant="outlined"
        startIcon={<AddIcon />}
        onClick={() => setForm({ ...form, etflist: [...form.etflist, ""] })}
        sx={{ alignSelf: "flex-start" }}
      >
        Add ETF
      </Button>

      <line></line>

      {[1, 2, 3].map((i) => (
        <Grid container spacing={2} key={i}>
          {form.etflist.map((_, idx) => (
            <Grid item xs={6} sm={4} md={3} key={idx}>
              <TextField
                label={`Alloc${i} - ${form.etflist[idx] || `ETF ${idx + 1}`}`}
                value={form[`allocation${i}`][idx] || ""}
                onChange={(e) => {
                  const updated = [...form[`allocation${i}`]];
                  updated[idx] = e.target.value;
                  setForm({ ...form, [`allocation${i}`]: updated });
                }}
                fullWidth
              />
            </Grid>
          ))}
        </Grid>
      ))}

      <Grid container spacing={2}>
        <Grid item xs={6}>
          <TextField
            type="date"
            label="Start Date"
            value={form.startdate}
            onChange={(e) => setForm({ ...form, startdate: e.target.value })}
            InputLabelProps={{ shrink: true }}
            fullWidth
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            type="date"
            label="End Date"
            value={form.enddate}
            onChange={(e) => setForm({ ...form, enddate: e.target.value })}
            InputLabelProps={{ shrink: true }}
            fullWidth
          />
        </Grid>
      </Grid>

      <Grid container spacing={2}>
        <Grid item xs={12} sm={6} md={4}>
            <TextField
                select
                label="Benchmark"
                value={form.benchmark}
                onChange={(e) => setForm({ ...form, benchmark: e.target.value })}
                fullWidth
                sx={{ minWidth: 200 }}
            >
                {benchmarkOptions.map((opt) => (
                <MenuItem key={opt} value={opt}>
                    {opt}
                </MenuItem>
                ))}
            </TextField>
        </Grid>
        <Grid item xs={6}>
          <TextField
            label="Start Balance"
            value={form.start_balance}
            onChange={(e) => setForm({ ...form, start_balance: e.target.value })}
            fullWidth
          />
        </Grid>
      </Grid>

      <Button variant="contained" onClick={handleSubmit} disabled={loading}>
        {loading ? "Running Backtest..." : "Run Backtest"}
      </Button>
    </Stack>
  );
}
