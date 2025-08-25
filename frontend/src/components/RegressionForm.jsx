import React from "react";
import { Grid, TextField, MenuItem, Button } from "@mui/material";
import EtfListInput from "./EtfListInput";
import AddIcon from "@mui/icons-material/Add";

const modelOptions = ["CAPM", "FF3", "FF4", "FF5"];

export default function RegressionForm({ form, setForm, onSubmit, loading }) {

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  return (
    <>
      <EtfListInput
        etflist={form.etflist}
        setEtflist={(newList) => setForm({ ...form, etflist: newList })}
      />
      <Button
        variant="outlined"
        startIcon={<AddIcon />}
        onClick={() => setForm({ ...form, etflist: [...form.etflist, ""] })}
        sx={{ mb: 2, alignSelf: "flex-start" }}
      >
        Add ETF
      </Button>
      <Grid container spacing={2}>
        <Grid item xs={6} sm={4} md={3}>
          <TextField
            name="model"
            label="Model"
            value={form.model}
            onChange={handleChange}
            select
            fullWidth
          >
            {modelOptions.map((m) => (
              <MenuItem key={m} value={m}>
                {m}
              </MenuItem>
            ))}
          </TextField>
        </Grid>

        <Grid item xs={6} sm={4} md={3}>
          <TextField
            label="Start Date"
            type="date"
            value={form.start_date}
            onChange={(e) => setForm({ ...form, start_date: e.target.value })}
            InputLabelProps={{ shrink: true }}
            fullWidth
          />
        </Grid>
        <Grid item xs={6} sm={4} md={3}>
          <TextField
            label="End Date"
            type="date"
            value={form.end_date}
            onChange={(e) => setForm({ ...form, end_date: e.target.value })}
            InputLabelProps={{ shrink: true }}
            fullWidth
          />
        </Grid>
        <Grid item xs={4}>
          <TextField
            label="Rolling Period (months)"
            type="number"
            value={form.rolling_period || 36}
            onChange={(e) =>
              setForm({ ...form, rolling_period: parseInt(e.target.value) })
            }
            fullWidth
          />
        </Grid>

        <Grid item xs={12} sm={4} md={3}>
          <Button
            variant="contained"
            onClick={onSubmit}
            fullWidth
            disabled={loading}
          >
            Run Regression
          </Button>
        </Grid>
      </Grid>
    </>
  );
}
