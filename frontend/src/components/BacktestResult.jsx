import React, { useMemo } from "react";
import {
  Box,
  Typography,
  Grid,
  Paper,
  Alert,
  Stack,
} from "@mui/material";
import Plot from 'react-plotly.js'; // Import Plotly
import DataTable from "./DataTable";

export default function BacktestResult({ result }) {
  const hasResult = Boolean(result);
  const safeResult = result ?? {};

  const buildFilename = (name, suffix) => {
    const base = (name || "portfolio").toString().trim();
    const sanitized = base.replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "").toLowerCase();
    return `${sanitized || "portfolio"}_${suffix}.csv`;
  };

  const formatValue = (value, digits = 4) => {
    if (value === null || value === undefined) return "N/A";
    if (typeof value === "number") return value.toFixed(digits);
    return value;
  };

  // --- Prepare data for Plotly charts ---

  // 1. Portfolio Growth Plot
  const portfolioGrowthData = Array.isArray(safeResult.portfolio_growth_plot_data)
    ? safeResult.portfolio_growth_plot_data
    : [];

  const portfolioGrowthTraces = portfolioGrowthData.map((series) => ({
      type: 'scatter',
      mode: 'lines',
      name: series.name,
      x: series.dates, // Expects array of date strings 'YYYY-MM-DD'
      y: series.values, // Expects array of numbers
    }));

  // 2. Annual Returns Plot (Grouped Bar Chart)
  const annualReturnsData = safeResult.annual_returns_plot_data ?? {};
  const annualReturnYears = Array.isArray(annualReturnsData.years) ? annualReturnsData.years : [];
  const annualReturnTraces = Array.isArray(annualReturnsData.series)
    ? annualReturnsData.series.map(series => ({
        type: 'bar',
        name: series.name,
        x: annualReturnYears, // Uses the common 'years' array
        y: series.returns,    // Expects array of numbers (percentages)
      }))
    : [];

  // 3. Drawdown Plot
  const drawdownData = Array.isArray(safeResult.drawdown_plot_data)
    ? safeResult.drawdown_plot_data
    : [];

  const drawdownTraces = drawdownData.map(series => ({
        type: 'scatter',
        mode: 'lines',
        name: series.name,
        x: series.dates, // Expects array of date strings 'YYYY-MM-DD'
        y: series.values, // Expects array of numbers (percentages)
      }));

  const infoMessages = Array.isArray(safeResult.messages) ? safeResult.messages : [];
  const warningMessages = Array.isArray(safeResult.warnings) ? safeResult.warnings : [];

  const allocationColumns = useMemo(
    () => [
      { accessorKey: "ticker", header: "Ticker" },
      {
        accessorKey: "allocation",
        header: "Allocation (%)",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
    ],
    [],
  );

  const summaryTable = Array.isArray(safeResult.summary_table) ? safeResult.summary_table : [];

  const summaryData = useMemo(() => {
    if (summaryTable.length === 0) return [];

    return summaryTable.map((row) => {
      const formattedRow = {};
      Object.entries(row).forEach(([key, value]) => {
        formattedRow[key] = formatValue(value);
      });
      return formattedRow;
    });
  }, [summaryTable]);

  const summaryColumns = useMemo(() => {
    if (summaryData.length === 0) return [];

    return Object.keys(summaryData[0]).map((key) => ({
      accessorKey: key,
      header: key,
      muiTableHeadCellProps: { align: key === "Metric" ? "left" : "right" },
      muiTableBodyCellProps: { align: key === "Metric" ? "left" : "right" },
    }));
  }, [summaryData]);

  const regressionColumns = useMemo(
    () => [
      { accessorKey: "factor", header: "Factor" },
      {
        accessorKey: "loadings",
        header: "Loadings",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
      {
        accessorKey: "stdErrors",
        header: "Std. Errors",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
      {
        accessorKey: "tStat",
        header: "t-stat",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
      {
        accessorKey: "pValue",
        header: "p-value",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
    ],
    [],
  );

  if (!hasResult) {
    return null;
  }

  return (
    <Box mt={4}>
      <Typography variant="h5">Backtest Output</Typography>

      {(infoMessages.length > 0 || warningMessages.length > 0) && (
        <Box mt={2}>
          <Stack spacing={1}>
            {infoMessages.map((message, idx) => (
              <Alert severity="info" key={`info-${idx}`}>
                {message}
              </Alert>
            ))}
            {warningMessages.map((message, idx) => (
              <Alert severity="warning" key={`warning-${idx}`}>
                {message}
              </Alert>
            ))}
          </Stack>
        </Box>
      )}

      {/* {safeResult.output_text && (
        <Paper elevation={3} sx={{ padding: 2, marginY: 2, whiteSpace: "pre-line", fontFamily: "monospace" }}>
          {safeResult.output_text}
        </Paper>
      )} */}

      {/* Portfolio Growth Plot */}
      {portfolioGrowthTraces.length > 0 && (
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Portfolio Growth
          </Typography>
          <Paper sx={{ p: 2 }}>
            <Plot
              data={portfolioGrowthTraces}
              layout={{
                title: 'Portfolio Growth',
                xaxis: { title: 'Date', type: 'date' },
                yaxis: { title: 'Portfolio Value' },
                autosize: true,
                height: 400,
                 margin: { t: 40, b: 80, l: 70, r: 30 } // Adjusted margins
              }}
              style={{ width: '100%'}}
              useResizeHandler={true}
            />
          </Paper>
        </Box>
      )}

      {/* Annual Returns Plot */}
      {annualReturnTraces.length > 0 && annualReturnYears.length > 0 && (
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Annual Returns
          </Typography>
          <Paper sx={{ p: 2 }}>
            <Plot
              data={annualReturnTraces}
              layout={{
                title: 'Annual Returns by Portfolio',
                xaxis: { title: 'Year', type: 'category' }, // Years as categories for bar chart
                yaxis: { title: 'Annual Return (%)' },
                barmode: 'group',
                autosize: true,
                height: 400,
                margin: { t: 40, b: 40, l: 60, r: 20 }
              }}
              style={{ width: '100%'}}
              useResizeHandler={true}
            />
          </Paper>
        </Box>
      )}

      {/* Drawdown Plot */}
      {drawdownTraces.length > 0 && (
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Portfolio Drawdowns
          </Typography>
          <Paper sx={{ p: 2 }}>
            <Plot
              data={drawdownTraces}
              layout={{
                title: 'Portfolio Drawdowns',
                xaxis: { title: 'Date', type: 'date' },
                yaxis: { title: 'Drawdown (%)' },
                autosize: true,
                height: 400,
                margin: { t: 40, b: 80, l: 70, r: 30 } // Adjusted margins
              }}
              style={{ width: '100%'}}
              useResizeHandler={true}
            />
          </Paper>
        </Box>
      )}

      {/* Backend-generated images (optional, if you still have some) */}
      {Array.isArray(safeResult.image_urls) && safeResult.image_urls.length > 0 && (
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Other Charts (from backend)
          </Typography>
          <Grid container spacing={2}>
            {safeResult.image_urls.map((url, idx) => (
              <Grid item xs={12} md={6} key={`img-${idx}`}>
                <img
                  src={`${process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000'}${url}?t=${Date.now()}`}
                  alt={`Chart ${idx}`}
                  style={{ width: "100%", border: "1px solid #ddd" }}
                />
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Portfolio Allocations Table */}
      {Array.isArray(safeResult.portfolio_allocations) && safeResult.portfolio_allocations.length > 0 && (
        safeResult.portfolio_allocations.map((portfolio, pIdx) => (
          portfolio.allocations && portfolio.allocations.length > 0 && (
            <Box key={`alloc-table-${pIdx}`} mt={3}>
              <DataTable
                title={`${portfolio.portfolioName} Allocations`}
                columns={allocationColumns}
                data={portfolio.allocations.map((alloc) => ({
                  ticker: alloc.ticker,
                  allocation: formatValue(alloc.Allocation, 2),
                }))}
                exportFileName={buildFilename(portfolio.portfolioName, "allocations")}
              />
            </Box>
          )
        ))
      )}

      {/* Performance Summary Table */}
      {summaryData.length > 0 && (
        <Box mt={4}>
          <DataTable
            title="Performance Summary"
            titleVariant="h6"
            columns={summaryColumns}
            data={summaryData}
            exportFileName="performance_summary"
          />
        </Box>
      )}

      {/* Drawdown Tables */}
      {Array.isArray(safeResult.drawdown_tables) && safeResult.drawdown_tables.length > 0 && (
        safeResult.drawdown_tables.map((tableData, pIdx) => (
          tableData.data && tableData.data.length > 0 && (
            <Box key={`drawdown-detail-table-${pIdx}`} mt={3}>
              <DataTable
                title={`Top 3 Drawdowns: ${tableData.portfolioName}`}
                columns={Object.keys(tableData.data[0] || {}).map((col) => ({
                  accessorKey: col,
                  header: col,
                  muiTableHeadCellProps: { align: typeof tableData.data[0][col] === 'number' ? "right" : "left" },
                  muiTableBodyCellProps: { align: typeof tableData.data[0][col] === 'number' ? "right" : "left" },
                }))}
                data={tableData.data.map((row) => {
                  const formattedRow = {};
                  Object.entries(row).forEach(([key, value]) => {
                    formattedRow[key] = formatValue(value);
                  });
                  return formattedRow;
                })}
                exportFileName={buildFilename(tableData.portfolioName, "drawdowns")}
              />
            </Box>
          )
        ))
      )}

      {/* Regression Analysis Tables */}
      {Array.isArray(safeResult.regression_table) && safeResult.regression_table.length > 0 && (
         safeResult.regression_table.map((regData, pIdx) => (
         regData.coefficients && regData.coefficients.length > 0 && (
            <Box key={`reg-summary-table-${pIdx}`} mt={3}>
              <DataTable
                title={`Regression Analysis vs Benchmark: ${regData.portfolioName}`}
                columns={regressionColumns}
                data={regData.coefficients.map((row) => ({
                  factor: row.Factor,
                  loadings: formatValue(row.Loadings),
                  stdErrors: formatValue(row['Standard Errors']),
                  tStat: formatValue(row['t-stat']),
                  pValue: formatValue(row['p-value']),
                }))}
                exportFileName={buildFilename(regData.portfolioName, "regression_coefficients")}
              />
              <Typography variant="body2" sx={{mb:1}}>
                  R-squared: {formatValue(regData.r_squared)}, Adj. R-squared: {formatValue(regData.adj_r_squared)}, Annualized Alpha: {formatValue(regData.annualized_alpha)}
              </Typography>
            </Box>
         )
         ))
      )}
    </Box>
  );
}
