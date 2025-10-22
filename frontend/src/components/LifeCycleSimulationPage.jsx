import { useCallback, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Divider,
  Grid,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import DownloadIcon from "@mui/icons-material/Download";
import Plot from "react-plotly.js";
import axios from "axios";

import DataTable from "./DataTable";
import { downloadCsvContent, escapeCsvValue } from "../utils/csv";

const formatNumber = (value, options = {}) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const formatter = new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 0,
    ...options,
  });
  return formatter.format(Number(value));
};

export default function LifeCycleSimulationPage() {
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";

  const [returnsFile, setReturnsFile] = useState(null);
  const [cashflowsFile, setCashflowsFile] = useState(null);
  const [initialWealth, setInitialWealth] = useState("0");
  const [minWealthCutoff, setMinWealthCutoff] = useState("0");
  const [nsim, setNsim] = useState("1000");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (setter) => (event) => {
    const file = event.target.files?.[0] ?? null;
    setter(file);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!returnsFile || !cashflowsFile) {
      setError("Please upload both return and cash flow CSV files.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("returns_file", returnsFile);
      formData.append("cashflows_file", cashflowsFile);
      formData.append("initial_wealth", initialWealth ?? "0");
      formData.append("wmin_cutoff", minWealthCutoff ?? "0");
      formData.append("nsim", nsim ?? "1000");

      const response = await axios.post(`${apiBaseUrl}/life-cycle/run`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data);
    } catch (err) {
      const message = err?.response?.data?.error || err?.message || "Failed to run life-cycle simulation.";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const histogramPlot = useMemo(() => {
    const histogram = result?.final_wealth_histogram;
    if (!histogram || !Array.isArray(histogram?.counts) || !Array.isArray(histogram?.bins)) {
      return null;
    }
    const counts = histogram.counts;
    const bins = histogram.bins;
    if (!counts.length || bins.length !== counts.length + 1) {
      return null;
    }

    const centers = counts.map((_, index) => (bins[index] + bins[index + 1]) / 2);
    const widths = counts.map((_, index) => bins[index + 1] - bins[index]);

    return {
      data: [
        {
          type: "bar",
          x: centers,
          y: counts,
          width: widths,
          marker: { color: "#1976d2" },
        },
      ],
      layout: {
        title: "Histogram of Final Wealth",
        xaxis: { title: "Final Period Wealth" },
        yaxis: { title: "Count" },
        bargap: 0.05,
        margin: { t: 60, r: 20, b: 60, l: 60 },
      },
    };
  }, [result]);

  const belowCutoffPlot = useMemo(() => {
    const belowCutoff = result?.wealth_below_cutoff;
    if (!belowCutoff || !Array.isArray(belowCutoff?.counts) || !Array.isArray(belowCutoff?.periods)) {
      return null;
    }

    return {
      data: [
        {
          type: "bar",
          x: belowCutoff.periods,
          y: belowCutoff.counts,
          marker: { color: "#388e3c" },
        },
      ],
      layout: {
        title: "Count of Wealth Below Cutoff by Period",
        xaxis: { title: "Period" },
        yaxis: { title: "Count" },
        margin: { t: 60, r: 20, b: 60, l: 60 },
      },
    };
  }, [result]);

  const histogramColumns = useMemo(
    () => [
      {
        accessorKey: "from",
        header: "From",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
        Cell: ({ value }) => formatNumber(value, { maximumFractionDigits: 0 }),
      },
      {
        accessorKey: "to",
        header: "To",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
        Cell: ({ value }) => formatNumber(value, { maximumFractionDigits: 0 }),
      },
      {
        accessorKey: "count",
        header: "Count",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
    ],
    [],
  );

  const belowCutoffColumns = useMemo(
    () => [
      {
        accessorKey: "period",
        header: "Period",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
      {
        accessorKey: "count",
        header: "Count",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
    ],
    [],
  );

  const summaryItems = useMemo(() => {
    const summary = result?.summary;
    if (!summary) {
      return [];
    }

    return [
      { label: "Mean final wealth", value: formatNumber(summary.mean) },
      { label: "Median final wealth", value: formatNumber(summary.median) },
      { label: "Minimum final wealth", value: formatNumber(summary.min) },
      { label: "Maximum final wealth", value: formatNumber(summary.max) },
      summary.std !== undefined
        ? { label: "Std. dev. final wealth", value: formatNumber(summary.std) }
        : null,
      {
        label: "Final count below cutoff",
        value: `${formatNumber(summary.final_below_cutoff_count, { maximumFractionDigits: 0 })}`,
      },
      {
        label: "Final % below cutoff",
        value: `${formatNumber(summary.final_below_cutoff_pct * 100, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}%`,
      },
      { label: "Cutoff", value: formatNumber(summary.wmin_cutoff) },
    ].filter(Boolean);
  }, [result]);

  const summaryDownloadRows = useMemo(() => {
    const summary = result?.summary;
    if (!summary) {
      return [];
    }

    const rows = [
      { metric: "Mean final wealth", value: formatNumber(summary.mean) },
      { metric: "Median final wealth", value: formatNumber(summary.median) },
      { metric: "Minimum final wealth", value: formatNumber(summary.min) },
      { metric: "Maximum final wealth", value: formatNumber(summary.max) },
    ];

    if (summary.std !== undefined) {
      rows.push({ metric: "Std. dev. final wealth", value: formatNumber(summary.std) });
    }

    rows.push(
      {
        metric: "Final count below cutoff",
        value: formatNumber(summary.final_below_cutoff_count, { maximumFractionDigits: 0 }),
      },
      {
        metric: "Final % below cutoff",
        value: `${formatNumber(summary.final_below_cutoff_pct * 100, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}%`,
      },
      { metric: "Cutoff", value: formatNumber(summary.wmin_cutoff) },
    );

    const metadata = result?.metadata;
    if (metadata) {
      rows.push(
        { metric: "Initial wealth", value: formatNumber(metadata.initial_wealth) },
        {
          metric: "Number of periods",
          value: formatNumber(metadata.n_periods, { maximumFractionDigits: 0 }),
        },
        { metric: "Simulations", value: formatNumber(metadata.n_simulations, { maximumFractionDigits: 0 }) },
      );
    }

    return rows;
  }, [result]);

  const summaryCsvContent = useMemo(() => {
    if (!summaryDownloadRows.length) {
      return null;
    }

    const header = "Metric,Value";
    const lines = summaryDownloadRows.map(({ metric, value }) =>
      `${escapeCsvValue(metric)},${escapeCsvValue(value)}`,
    );

    return [header, ...lines].join("\n");
  }, [summaryDownloadRows]);

  const handleDownloadSummary = useCallback(() => {
    if (!summaryCsvContent) {
      return;
    }

    downloadCsvContent(summaryCsvContent, "life-cycle-summary.csv");
  }, [summaryCsvContent]);

  const metadataItems = useMemo(() => {
    const metadata = result?.metadata;
    if (!metadata) {
      return [];
    }
    return [
      { label: "Initial wealth", value: formatNumber(metadata.initial_wealth) },
      { label: "Number of periods", value: formatNumber(metadata.n_periods, { maximumFractionDigits: 0 }) },
      { label: "Simulations", value: formatNumber(metadata.n_simulations, { maximumFractionDigits: 0 }) },
    ];
  }, [result]);

  const resetForm = () => {
    setReturnsFile(null);
    setCashflowsFile(null);
    setInitialWealth("0");
    setMinWealthCutoff("0");
    setNsim("1000");
    setResult(null);
    setError(null);
  };

  return (
    <Box component="form" onSubmit={handleSubmit} noValidate>
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          Life-Cycle Simulation
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Upload return and cash flow series to simulate wealth paths across the life cycle. Adjust the
          initial wealth, minimum wealth cutoff, and number of simulations to tailor the analysis.
        </Typography>

        {error ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        ) : null}

        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} md={6}>
            <Button
              component="label"
              variant="outlined"
              startIcon={<UploadFileIcon />}
              fullWidth
            >
              {returnsFile ? `Return file: ${returnsFile.name}` : "Upload Returns CSV"}
              <input type="file" hidden accept=".csv" onChange={handleFileChange(setReturnsFile)} />
            </Button>
          </Grid>
          <Grid item xs={12} md={6}>
            <Button
              component="label"
              variant="outlined"
              startIcon={<UploadFileIcon />}
              fullWidth
            >
              {cashflowsFile ? `Cash flow file: ${cashflowsFile.name}` : "Upload Cash Flow CSV"}
              <input type="file" hidden accept=".csv" onChange={handleFileChange(setCashflowsFile)} />
            </Button>
          </Grid>
        </Grid>

        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={12} md={4}>
            <TextField
              label="Initial wealth"
              value={initialWealth}
              onChange={(event) => setInitialWealth(event.target.value)}
              fullWidth
              size="small"
              inputProps={{ inputMode: "decimal" }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              label="Minimum wealth cutoff"
              value={minWealthCutoff}
              onChange={(event) => setMinWealthCutoff(event.target.value)}
              fullWidth
              size="small"
              inputProps={{ inputMode: "decimal" }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              label="Simulations"
              value={nsim}
              onChange={(event) => setNsim(event.target.value)}
              fullWidth
              size="small"
              inputProps={{ inputMode: "numeric" }}
            />
          </Grid>
        </Grid>

        <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
          <LoadingButton type="submit" variant="contained" loading={loading}>
            Run Simulation
          </LoadingButton>
          <Button variant="outlined" color="secondary" onClick={resetForm} disabled={loading}>
            Reset
          </Button>
        </Stack>
      </Paper>

      {result ? (
        <Stack spacing={4}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Stack
              direction={{ xs: "column", sm: "row" }}
              spacing={2}
              alignItems={{ xs: "flex-start", sm: "center" }}
              justifyContent="space-between"
              sx={{ mb: 2 }}
            >
              <Typography variant="h6">Summary</Typography>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleDownloadSummary}
                disabled={!summaryCsvContent}
              >
                Download Summary
              </Button>
            </Stack>
            <Grid container spacing={2}>
              {summaryItems.map((item) => (
                <Grid item xs={12} sm={6} md={4} key={item.label}>
                  <Typography variant="subtitle2" color="text.secondary">
                    {item.label}
                  </Typography>
                  <Typography variant="body1">{item.value}</Typography>
                </Grid>
              ))}
              {metadataItems.length ? (
                <Grid item xs={12}>
                  <Divider sx={{ my: 1 }} />
                </Grid>
              ) : null}
              {metadataItems.map((item) => (
                <Grid item xs={12} sm={6} md={4} key={item.label}>
                  <Typography variant="subtitle2" color="text.secondary">
                    {item.label}
                  </Typography>
                  <Typography variant="body1">{item.value}</Typography>
                </Grid>
              ))}
            </Grid>
          </Paper>

          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Final Wealth Distribution
            </Typography>
            {histogramPlot ? (
              <Plot data={histogramPlot.data} layout={histogramPlot.layout} style={{ width: "100%" }} />
            ) : (
              <Typography variant="body2" color="text.secondary">
                Histogram data is unavailable.
              </Typography>
            )}
            <DataTable
              title="Final Period Wealth Distribution"
              columns={histogramColumns}
              data={result?.final_wealth_histogram?.table ?? []}
              exportFileName="final-wealth-histogram.csv"
            />
          </Paper>

          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Wealth Below Cutoff
            </Typography>
            {belowCutoffPlot ? (
              <Plot data={belowCutoffPlot.data} layout={belowCutoffPlot.layout} style={{ width: "100%" }} />
            ) : (
              <Typography variant="body2" color="text.secondary">
                Cutoff data is unavailable.
              </Typography>
            )}
            <DataTable
              title="Count of Wealth Below Cutoff by Period"
              columns={belowCutoffColumns}
              data={result?.wealth_below_cutoff?.table ?? []}
              exportFileName="wealth-below-cutoff.csv"
            />
          </Paper>
        </Stack>
      ) : null}
    </Box>
  );
}

