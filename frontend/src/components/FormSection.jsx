import { useState } from "react";
import {
  Alert,
  Button,
  TextField,
  Paper,
  Stack,
  Typography,
  FormControlLabel,
  Checkbox,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import axios from "axios";
import dayjs from "dayjs";
import { LoadingButton } from "@mui/lab";

import EtfListInput from "./EtfListInput";
import AddIcon from "@mui/icons-material/Add";

export default function FormSection({ setResult }) {
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";
  const earliestDate = dayjs("1993-02-01");
  const latestDate = dayjs("2023-12-31");

  const [form, setForm] = useState({
    short: false,
    mvType: "standard",
    startdate: "2020-01-01",
    enddate: "2023-12-31",
    etflist: [""],
  });

  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState([]);

  const validateForm = () => {
    const validationErrors = [];
    const tickers = form.etflist.map((ticker) => ticker?.trim().toUpperCase() || "");

    const start = dayjs(form.startdate);
    const end = dayjs(form.enddate);

    if (!start.isValid()) {
      validationErrors.push("Start date is invalid.");
    }

    if (!end.isValid()) {
      validationErrors.push("End date is invalid.");
    }

    if (start.isValid() && end.isValid()) {
      if (start.isAfter(end)) {
        validationErrors.push("Start date must be before or equal to the end date.");
      }

      if (start.isBefore(earliestDate)) {
        validationErrors.push("Start date is earlier than the available data range (1993-02).");
      }

      if (end.isAfter(latestDate)) {
        validationErrors.push("End date is later than the available data range (2023-12).");
      }
    }

    const activeTickers = tickers.filter((ticker) => ticker !== "");
    if (activeTickers.length === 0) {
      validationErrors.push("Please provide at least one ETF ticker.");
    }

    const seen = new Set();
    const duplicates = new Set();
    activeTickers.forEach((ticker) => {
      if (seen.has(ticker)) {
        duplicates.add(ticker);
      } else {
        seen.add(ticker);
      }
    });
    if (duplicates.size > 0) {
      validationErrors.push(
        `Duplicate tickers detected: ${Array.from(duplicates).join(", ")}. Remove duplicates before running the analysis.`
      );
    }

    const invalidTickers = activeTickers.filter((ticker) => !/^[A-Z0-9\.\-]+$/.test(ticker));
    if (invalidTickers.length > 0) {
      validationErrors.push(
        `Tickers may only contain letters, numbers, periods, or dashes: ${invalidTickers.join(", ")}.`
      );
    }

    return validationErrors;
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === "checkbox" ? checked : value;
    setForm((prev) => ({
      ...prev,
      [name]: newValue,
      ...(name === "mvType" && newValue === "robust" ? { short: false } : {}),
    }));
  };

  const handleSubmit = async () => {
    const validationErrors = validateForm();
    if (validationErrors.length > 0) {
      setErrors(validationErrors);
      setResult(null);
      return;
    }

    setErrors([]);
    setResult(null);
    setLoading(true);
    try {
      const uniqueTickers = [];
      const seenTickers = new Set();
      form.etflist.forEach((ticker) => {
        const cleaned = ticker?.trim().toUpperCase();
        if (cleaned && !seenTickers.has(cleaned)) {
          seenTickers.add(cleaned);
          uniqueTickers.push(cleaned);
        }
      });

      const payload = {
        etflist: uniqueTickers.join(","),
        short: form.short ? 1 : 0,
        maxuse: 0,
        normal: form.mvType === "standard" ? 1 : 0,
        startdate: dayjs(form.startdate).format("YYYYMM"),
        enddate: dayjs(form.enddate).format("YYYYMM"),
      };

      const res = await axios.post(`${apiBaseUrl}/run`, payload);
      setResult(res.data);
    } catch (err) {
      console.error("Submission error:", err);
      const responseErrors =
        err.response?.data?.errors ||
        (err.response?.data?.error ? [err.response.data.error] : ["Failed to run mean-variance analysis. Please try again."]);
      setErrors(responseErrors);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };


  return (
    <Paper elevation={3} sx={{ padding: 4 }}>
      <Typography variant="h6" gutterBottom>
        Portfolio Parameters
      </Typography>
      <Stack spacing={2}>
        {errors.map((error, idx) => (
          <Alert severity="error" key={`mv-error-${idx}`}>
            {error}
          </Alert>
        ))}

        <Typography variant="subtitle1" gutterBottom>
          ETF List
        </Typography>

        <EtfListInput
          etflist={form.etflist}
          setEtflist={(newList) => setForm({ ...form, etflist: newList })}
          size={4}
        />

        <Button
          variant="outlined"
          startIcon={<AddIcon />}
          onClick={() => setForm({ ...form, etflist: [...form.etflist, ""] })}
          sx={{ alignSelf: "flex-start" }}
        >
          Add ETF
        </Button>

        <Stack direction="row" spacing={2}>
          <TextField
            name="startdate"
            label="Start Date"
            type="date"
            value={form.startdate}
            onChange={handleChange}
            InputLabelProps={{ shrink: true }}
            fullWidth
          />
          <TextField
            name="enddate"
            label="End Date"
            type="date"
            value={form.enddate}
            onChange={handleChange}
            InputLabelProps={{ shrink: true }}
            fullWidth
          />
        </Stack>

        <FormControl fullWidth>
          <InputLabel id="mvType-label">Mean-Variance Type</InputLabel>
          <Select
            labelId="mvType-label"
            name="mvType"
            value={form.mvType}
            label="Mean-Variance Type"
            onChange={handleChange}
          >
            <MenuItem value="standard">Standard Mean-Variance</MenuItem>
            <MenuItem value="robust">Robust Mean-Variance</MenuItem>
          </Select>
        </FormControl>
        {form.mvType === "standard" && (
          <FormControlLabel
            control={
              <Checkbox name="short" checked={form.short} onChange={handleChange} />
            }
            label="Allow Short"
          />
        )}

        <LoadingButton
          variant="contained"
          onClick={handleSubmit}
          loading={loading}
        >
          Run Optimization
        </LoadingButton>

      </Stack>
    </Paper>
  );
}
