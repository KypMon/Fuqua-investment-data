import { useState } from "react";
import {
  Button,
  TextField,
  Paper,
  Stack,
  Typography,
  FormControlLabel,
  Checkbox,
} from "@mui/material";
import axios from "axios";
import dayjs from "dayjs";
import { LoadingButton } from "@mui/lab";

import EtfListInput from "./EtfListInput";

export default function FormSection({ setResult }) {

  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL;

  const [form, setForm] = useState({
    short: false,
    maxuse: false,
    normal: false,
    startdate: "2020-01-01",
    enddate: "2023-12-31",
    etflist: [],
  });

  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm({ ...form, [name]: type === "checkbox" ? checked : value });
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append(
        "etflist",
        form.etflist.filter((e) => e.trim() !== "").join(",")
      );
      

      formData.append("short", form.short ? 1 : 0);
      formData.append("maxuse", form.maxuse ? 1 : 0);
      formData.append("normal", form.normal ? 1 : 0);
      formData.append("startdate", dayjs(form.startdate).format("YYYYMM"));
      formData.append("enddate", dayjs(form.enddate).format("YYYYMM"));
  
      const res = await axios.post(`${apiBaseUrl}/run`, formData);
      setResult(res.data);
    } catch (err) {
      console.error("Submission error:", err);
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

      <EtfListInput
        etflist={form.etflist}
        setEtflist={(newList) => setForm({ ...form, etflist: newList })}
        />

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


        <FormControlLabel
          control={
            <Checkbox name="short" checked={form.short} onChange={handleChange} />
          }
          label="Allow Short"
        />
        <FormControlLabel
          control={
            <Checkbox name="maxuse" checked={form.maxuse} onChange={handleChange} />
          }
          label="Max Use Constraint"
        />
        <FormControlLabel
          control={
            <Checkbox name="normal" checked={form.normal} onChange={handleChange} />
          }
          label="Use Normal Distribution"
        />

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
