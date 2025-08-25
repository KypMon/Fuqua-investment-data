// src/components/BacktestForm.jsx
import React, { useState } from "react";
import { Button, Stack, TextField, Grid, MenuItem } from "@mui/material";
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

  const handleSubmit = async () => {
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
    } catch (err) {
      console.error("Backtest error:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Stack spacing={3}>
      <EtfListInput
        etflist={form.etflist}
        setEtflist={(list) => setForm({ ...form, etflist: list })}
      />

      <Button
        variant="outlined"
        startIcon={<AddIcon />}
        onClick={() => setForm({ ...form, etflist: [...form.etflist, ""] })}
        sx={{ alignSelf: "flex-start" }}
      >
        Add ETF
      </Button>

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
