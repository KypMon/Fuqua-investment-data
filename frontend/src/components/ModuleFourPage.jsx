import { useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Divider,
  Grid,
  Link,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { LoadingButton } from "@mui/lab";
import AddIcon from "@mui/icons-material/Add";
import Plot from "react-plotly.js";
import axios from "axios";

import DataTable from "./DataTable";
import EtfListInput from "./EtfListInput";

const DEFAULT_TICKERS = ["SPY", "IWM", "TLT", "LQD", "IEF"];
const DEFAULT_START = "2020-12-31";
const DEFAULT_END = "2023-12-31";

const formatNumber = (value, digits = 4) => {
  if (value === null || value === undefined) {
    return "";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return value;
  }
  return numeric.toFixed(digits);
};

const normalizeColumnList = (source) => {
  if (Array.isArray(source)) {
    return source;
  }

  if (Array.isArray(source?.columns)) {
    return source.columns;
  }

  if (Array.isArray(source?.records) && source.records.length > 0) {
    return Object.keys(source.records[0]);
  }

  return [];
};

const createColumns = (source = [], decimals = 6) =>
  normalizeColumnList(source).map((column) => {
    const key = typeof column === "string" ? column : String(column);
    const lowerKey = key.toLowerCase();
    const isDate = lowerKey === "date";
    const isLabel = lowerKey === "assets" || lowerKey === "asset";
    const align = isDate ? "left" : "right";

    return {
      accessorKey: key,
      header: key,
      muiTableHeadCellProps: { align: isLabel ? "left" : align },
      muiTableBodyCellProps: { align: isLabel ? "left" : align },
      Cell: ({ value }) => {
        if (value === null || value === undefined || value === "") {
          return "";
        }
        if (isDate) {
          return String(value);
        }
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
          return value;
        }
        return numeric.toFixed(decimals);
      },
    };
  });

const buildPortfolioTable = (labelA, labelB, blockA, blockB, assets = []) => {
  const baseColumns = [
    { accessorKey: "label", header: "", muiTableHeadCellProps: { align: "left" } },
    {
      accessorKey: "sigma",
      header: "sigma[r]",
      muiTableHeadCellProps: { align: "right" },
      muiTableBodyCellProps: { align: "right" },
      Cell: ({ value }) => formatNumber(value, 4),
    },
    {
      accessorKey: "mean",
      header: "E[r]",
      muiTableHeadCellProps: { align: "right" },
      muiTableBodyCellProps: { align: "right" },
      Cell: ({ value }) => formatNumber(value, 4),
    },
  ];

  const weightColumns = assets.map((asset) => ({
    accessorKey: asset,
    header: `w(${asset})`,
    muiTableHeadCellProps: { align: "right" },
    muiTableBodyCellProps: { align: "right" },
    Cell: ({ value }) => formatNumber(value, 4),
  }));

  const toRow = (label, block) => {
    if (!block) {
      return { label };
    }

    const weights = block?.weights ?? {};
    const row = {
      label,
      sigma: block?.sigma ?? null,
      mean: block?.mean ?? null,
    };

    assets.forEach((asset) => {
      row[asset] = weights[asset] ?? null;
    });

    return row;
  };

  return {
    columns: [...baseColumns, ...weightColumns],
    data: [toRow(labelA, blockA), toRow(labelB, blockB)],
  };
};

const extractError = (error, fallback = "Unexpected error") => {
  if (error?.response?.data?.error) {
    return error.response.data.error;
  }
  if (error?.message) {
    return error.message;
  }
  return fallback;
};

export default function ModuleFourPage() {
  const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";

  const [tickers, setTickers] = useState(() => [...DEFAULT_TICKERS]);
  const sanitizedTickers = useMemo(
    () => tickers.map((ticker) => ticker.trim()).filter(Boolean),
    [tickers],
  );
  const hasMinimumTickers = sanitizedTickers.length >= 2;
  const [startDate, setStartDate] = useState(DEFAULT_START);
  const [endDate, setEndDate] = useState(DEFAULT_END);

  const [matretState, setMatretState] = useState(null);
  const [matretError, setMatretError] = useState(null);
  const [matretLoading, setMatretLoading] = useState(false);
  const [matretUploadLoading, setMatretUploadLoading] = useState(false);

  const [matErCovrState, setMatErCovrState] = useState(null);
  const [matErCovrError, setMatErCovrError] = useState(null);
  const [matErCovrLoading, setMatErCovrLoading] = useState(false);
  const [matErCovrUploadLoading, setMatErCovrUploadLoading] = useState(false);

  const [riskFree, setRiskFree] = useState("0.03");

  const [portfolioState, setPortfolioState] = useState(null);
  const [portfolioError, setPortfolioError] = useState(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);

  const handleGenerateMatret = async () => {
    if (!hasMinimumTickers) {
      setMatretError("Please provide at least two ticker symbols.");
      return;
    }

    setMatretError(null);
    setMatretLoading(true);
    try {
      const response = await axios.post(`${apiBaseUrl}/module4/matret/generate`, {
        tickers: sanitizedTickers,
        start_date: startDate,
        end_date: endDate,
      });

      setMatretState(response.data);
      setPortfolioState(null);
      if (matErCovrState) {
        setMatErCovrState(null);
      }

      if (Array.isArray(response.data?.tickers) && response.data.tickers.length > 0) {
        const nextTickers = response.data.tickers.map((value) =>
          value == null ? "" : String(value).trim(),
        );
        setTickers(nextTickers.length >= 2 ? nextTickers : [...nextTickers, ""]);
      }
    } catch (error) {
      setMatretError(extractError(error, "Failed to generate matret"));
    } finally {
      setMatretLoading(false);
    }
  };

  const handleUploadMatret = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setMatretError(null);
    setMatretUploadLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${apiBaseUrl}/module4/matret/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMatretState(response.data);
      setPortfolioState(null);
      if (matErCovrState) {
        setMatErCovrState(null);
      }
    } catch (error) {
      setMatretError(extractError(error, "Failed to upload matret"));
    } finally {
      setMatretUploadLoading(false);
      event.target.value = "";
    }
  };

  const handleGenerateMatErCovr = async () => {
    if (!matretState?.matrix) {
      setMatErCovrError("matret matrix is required");
      return;
    }

    const riskValue = riskFree === "" ? null : Number(riskFree);
    if (riskFree !== "" && !Number.isFinite(riskValue)) {
      setMatErCovrError("Risk-free rate must be numeric");
      return;
    }

    setMatErCovrError(null);
    setMatErCovrLoading(true);
    try {
      const response = await axios.post(`${apiBaseUrl}/module4/mat_er_covr/generate`, {
        matret: matretState.matrix,
        risk_free: riskValue,
      });
      setMatErCovrState(response.data);
      setPortfolioState(null);
      if (response.data?.risk_free !== undefined && response.data?.risk_free !== null) {
        setRiskFree(String(response.data.risk_free));
      }
    } catch (error) {
      setMatErCovrError(extractError(error, "Failed to create mat_er_covr"));
    } finally {
      setMatErCovrLoading(false);
    }
  };

  const handleUploadMatErCovr = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    setMatErCovrError(null);
    setMatErCovrUploadLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${apiBaseUrl}/module4/mat_er_covr/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMatErCovrState(response.data);
      setPortfolioState(null);
      if (response.data?.risk_free !== undefined && response.data?.risk_free !== null) {
        setRiskFree(String(response.data.risk_free));
      }
    } catch (error) {
      setMatErCovrError(extractError(error, "Failed to upload mat_er_covr"));
    } finally {
      setMatErCovrUploadLoading(false);
      event.target.value = "";
    }
  };

  const handleComputePortfolios = async () => {
    if (!matErCovrState?.matrix) {
      setPortfolioError("mat_er_covr matrix is required");
      return;
    }

    const riskValue = riskFree === "" ? null : Number(riskFree);
    if (riskFree !== "" && !Number.isFinite(riskValue)) {
      setPortfolioError("Risk-free rate must be numeric");
      return;
    }

    setPortfolioError(null);
    setPortfolioLoading(true);
    try {
      const response = await axios.post(`${apiBaseUrl}/module4/portfolios`, {
        mat_er_covr: matErCovrState.matrix,
        risk_free: riskValue,
      });
      setPortfolioState(response.data);
    } catch (error) {
      setPortfolioError(extractError(error, "Failed to compute portfolios"));
    } finally {
      setPortfolioLoading(false);
    }
  };

  const matretColumns = useMemo(() => createColumns(matretState?.matrix ?? []), [
    matretState,
  ]);

  const matretData = useMemo(
    () => matretState?.matrix?.records ?? [],
    [matretState],
  );

  const matretDownloadUrl = useMemo(() => {
    const path = matretState?.csv_url;
    if (!path) return null;
    return path.startsWith("http") ? path : `${apiBaseUrl}${path}`;
  }, [matretState, apiBaseUrl]);

  const matErCovrColumns = useMemo(
    () => createColumns(matErCovrState?.matrix ?? [], 6),
    [matErCovrState],
  );

  const matErCovrData = useMemo(
    () => matErCovrState?.matrix?.records ?? [],
    [matErCovrState],
  );

  const matErCovrDownloadUrl = useMemo(() => {
    const path = matErCovrState?.csv_url;
    if (!path) return null;
    return path.startsWith("http") ? path : `${apiBaseUrl}${path}`;
  }, [matErCovrState, apiBaseUrl]);

  const assets = portfolioState?.assets ?? [];
  const longOnly = portfolioState?.long_only ?? null;
  const shortAllowed = portfolioState?.short_allowed ?? null;

  const minVarTable = useMemo(
    () => buildPortfolioTable(
      "Without Short Sales",
      "With Short Sales",
      longOnly?.min_variance,
      shortAllowed?.min_variance,
      assets,
    ),
    [assets, longOnly, shortAllowed],
  );

  const tangencyTable = useMemo(
    () => buildPortfolioTable(
      "Without Short Sales",
      "With Short Sales",
      longOnly?.tangency,
      shortAllowed?.tangency,
      assets,
    ),
    [assets, longOnly, shortAllowed],
  );

  const frontierPlotData = useMemo(() => {
    if (!portfolioState) {
      return [];
    }

    const traces = [];

    if (Array.isArray(longOnly?.frontier) && longOnly.frontier.length > 0) {
      traces.push({
        x: longOnly.frontier.map((p) => p.x),
        y: longOnly.frontier.map((p) => p.y),
        mode: "lines",
        name: "Frontier (No Short)",
        line: { color: "#1976d2" },
      });
    }

    if (Array.isArray(shortAllowed?.frontier) && shortAllowed.frontier.length > 0) {
      traces.push({
        x: shortAllowed.frontier.map((p) => p.x),
        y: shortAllowed.frontier.map((p) => p.y),
        mode: "lines",
        name: "Frontier (Short Allowed)",
        line: { color: "#9c27b0", dash: "dash" },
      });
    }

    const addMarker = (label, point, color) => {
      if (!point) return;
      const { sigma, mean } = point;
      if (sigma === undefined || mean === undefined) {
        return;
      }
      traces.push({
        x: [sigma],
        y: [mean],
        mode: "markers+text",
        name: label,
        text: [label],
        textposition: "top center",
        marker: { color, size: 11, symbol: "star" },
      });
    };

    addMarker("Tangency (No Short)", longOnly?.tangency, "#ff9800");
    addMarker("Tangency (Short)", shortAllowed?.tangency, "#ff5722");
    addMarker("Min Var (No Short)", longOnly?.min_variance, "#2e7d32");
    addMarker("Min Var (Short)", shortAllowed?.min_variance, "#009688");

    return traces;
  }, [portfolioState, longOnly, shortAllowed]);

  return (
    <Stack spacing={4}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h6" gutterBottom>
          Step 1: Create or upload <code>matret</code>
        </Typography>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Generate from Yahoo! Finance</Typography>

          <EtfListInput
            etflist={tickers}
            setEtflist={setTickers}
            size={4}
            minItems={2}
          />
          
          <Grid container>
              <Stack spacing={4}>
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={() => setTickers((current) => [...current, ""])}
                  sx={{ alignSelf: "flex-start" }}
                >
                  Add Ticker
                </Button>
                <Typography variant="caption" color="text.secondary">
                  Enter at least two ticker symbols.
                </Typography>
              </Stack>
          </Grid>

          <Stack direction="row" spacing={2}>            
            <TextField
              label="Start Date"
              type="date"
              value={startDate}
              onChange={(event) => setStartDate(event.target.value)}
              InputLabelProps={{ shrink: true }}
              fullWidth
            />

            <TextField
              label="End Date"
              type="date"
              value={endDate}
              onChange={(event) => setEndDate(event.target.value)}
              InputLabelProps={{ shrink: true }}
              fullWidth
            />
          </Stack>

          <LoadingButton
            variant="contained"
            onClick={handleGenerateMatret}
            loading={matretLoading}
            disabled={!hasMinimumTickers}
            sx={{ alignSelf: "flex-start" }}
          >
            Generate matret
          </LoadingButton>
          <Divider flexItem>
            <Typography variant="body2">or</Typography>
          </Divider>
          <Stack direction="row" spacing={2} alignItems="center">
            <Button variant="outlined" component="label" disabled={matretUploadLoading}>
              Upload matret CSV
              <input type="file" accept=".csv" hidden onChange={handleUploadMatret} />
            </Button>
            {matretUploadLoading && <Typography variant="body2">Uploading…</Typography>}
          </Stack>
          {matretError && <Alert severity="error">{matretError}</Alert>}
          {matretDownloadUrl && (
            <Link href={matretDownloadUrl} target="_blank" rel="noopener">
              Download latest matret CSV
            </Link>
          )}
        </Stack>
      </Paper>

      {matretData.length > 0 && (
        <Paper elevation={1} sx={{ p: 4 }}>
          <Typography variant="subtitle1" gutterBottom>
            matret preview
          </Typography>
          <DataTable
            columns={matretColumns}
            data={matretData}
            title="matret"
            titleVariant="h6"
            exportFileName="matret"
          />
        </Paper>
      )}

      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h6" gutterBottom>
          Step 2: Create or upload <code>mat_er_covr</code>
        </Typography>
        <Stack spacing={2}>
          <Typography variant="subtitle1">Create from matret</Typography>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems="center">
            <TextField
              label="Risk-free rate"
              value={riskFree}
              onChange={(event) => setRiskFree(event.target.value)}
              helperText="Use decimal (e.g., 0.03)"
            />
            <LoadingButton
              variant="contained"
              onClick={handleGenerateMatErCovr}
              loading={matErCovrLoading}
              disabled={!matretState?.matrix}
            >
              Generate mat_er_covr
            </LoadingButton>
          </Stack>
          <Divider flexItem>
            <Typography variant="body2">or</Typography>
          </Divider>
          <Stack direction="row" spacing={2} alignItems="center">
            <Button variant="outlined" component="label" disabled={matErCovrUploadLoading}>
              Upload mat_er_covr CSV
              <input type="file" accept=".csv" hidden onChange={handleUploadMatErCovr} />
            </Button>
            {matErCovrUploadLoading && <Typography variant="body2">Uploading…</Typography>}
          </Stack>
          {matErCovrError && <Alert severity="error">{matErCovrError}</Alert>}
          {matErCovrDownloadUrl && (
            <Link href={matErCovrDownloadUrl} target="_blank" rel="noopener">
              Download latest mat_er_covr CSV
            </Link>
          )}
        </Stack>
      </Paper>

      {matErCovrData.length > 0 && (
        <Paper elevation={1} sx={{ p: 4 }}>
          <Typography variant="subtitle1" gutterBottom>
            mat_er_covr preview
          </Typography>
          <DataTable
            columns={matErCovrColumns}
            data={matErCovrData}
            title="mat_er_covr"
            titleVariant="h6"
            exportFileName="mat_er_covr"
          />
        </Paper>
      )}

      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h6" gutterBottom>
          Step 3: Compute efficient frontier and portfolios
        </Typography>
        <Stack spacing={2}>
          <TextField
            label="Risk-free rate"
            value={riskFree}
            onChange={(event) => setRiskFree(event.target.value)}
            helperText="Used for tangency portfolio"
            sx={{ maxWidth: 240 }}
          />
          <LoadingButton
            variant="contained"
            onClick={handleComputePortfolios}
            loading={portfolioLoading}
            disabled={!matErCovrState?.matrix}
            sx={{ alignSelf: "flex-start" }}
          >
            Compute portfolios
          </LoadingButton>
          {portfolioError && <Alert severity="error">{portfolioError}</Alert>}
        </Stack>
      </Paper>

      {portfolioState && (
        <Paper elevation={1} sx={{ p: 4 }}>
          <Typography variant="h6" gutterBottom>
            Mean-Standard Deviation Frontier
          </Typography>
          <Box sx={{ height: 420 }}>
            <Plot
              data={frontierPlotData}
              layout={{
                autosize: true,
                margin: { l: 60, r: 20, t: 30, b: 60 },
                xaxis: { title: "Standard deviation" },
                yaxis: { title: "Expected return" },
                legend: { orientation: "h", y: -0.2 },
              }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
            />
          </Box>

          <Box mt={4}>
            <Typography variant="subtitle1" gutterBottom>
              Minimum-Variance Portfolio (with/without short sales)
            </Typography>
            <DataTable
              columns={minVarTable.columns}
              data={minVarTable.data}
              exportFileName="min_variance_portfolios"
            />
          </Box>

          <Box mt={4}>
            <Typography variant="subtitle1" gutterBottom>
              Tangency Portfolio (with/without short sales)
            </Typography>
            <DataTable
              columns={tangencyTable.columns}
              data={tangencyTable.data}
              exportFileName="tangency_portfolios"
            />
          </Box>
        </Paper>
      )}
    </Stack>
  );
}

