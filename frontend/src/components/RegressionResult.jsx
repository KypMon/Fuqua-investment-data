import React, { useMemo } from "react";
import {
  Box,
  Typography,
  Paper,
  Grid,
  Alert,
  Stack
} from "@mui/material";
import Plot from 'react-plotly.js'; // Make sure react-plotly.js is installed
import OlsSummary from './OlsSummary';
import DataTable from "./DataTable";

export default function RegressionResult({ result }) {
  
  if (!result) return null;

  // result should already be an object if your axios request is correctly configured.
  // If it might still be a string, this parsing is fine.
  const resultData = typeof result === "string"
  ? JSON.parse(result)
  : result;

  // --- Prepare data for Rolling Alpha and Factor Loadings Plotly Chart ---
  let rollingPlotTraces = [];
  const rollingPlotLayout = {
    title: 'Rolling Alpha & Factor Loadings',
    xaxis: { title: 'Date', type: 'date' },
    yaxis: { 
      title: 'Annualized Alpha', 
      titlefont: { color: 'red' }, 
      tickfont: { color: 'red' } 
    },
    yaxis2: {
      title: 'Factor Loadings',
      titlefont: { color: 'blue' },
      tickfont: { color: 'blue' },
      overlaying: 'y',
      side: 'right'
    },
    autosize: true,
    height: 500,
    legend: { 
      x: 0.5, 
      y: -0.3, // Adjusted y to be below the chart
      xanchor: 'center', 
      orientation: 'h',
      traceorder: 'normal' // To control legend item order if needed
    },
    margin: { t: 40, b: 120, l: 70, r: 70 } // Increased bottom margin for legend
  };

  if (resultData.rolling_plot_data &&
      resultData.rolling_plot_data.dates &&
      resultData.rolling_plot_data.dates.length > 0) {
    
    // 1. Alpha Series (Primary Y-axis)
    if (resultData.rolling_plot_data.alpha_series) {
      rollingPlotTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Annualized Alpha',
        x: resultData.rolling_plot_data.dates,
        y: resultData.rolling_plot_data.alpha_series,
        yaxis: 'y1',
        line: { color: 'red' }
      });
    }

    // 2. Factor Loadings Series (Secondary Y-axis)
    if (Array.isArray(resultData.rolling_plot_data.factor_series)) {
      const plotlyLinestyles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'];
      resultData.rolling_plot_data.factor_series.forEach((series, index) => {
        rollingPlotTraces.push({
          type: 'scatter',
          mode: 'lines',
          name: series.name, // Factor name from backend data
          x: resultData.rolling_plot_data.dates,
          y: series.values,
          yaxis: 'y2',
          line: { dash: plotlyLinestyles[index % plotlyLinestyles.length] }
        });
      });
    }
  }

  const errorMessages = Array.isArray(resultData.errors) ? resultData.errors : [];
  const infoMessages = Array.isArray(resultData.messages) ? resultData.messages : [];
  const warningMessages = Array.isArray(resultData.warnings) ? resultData.warnings : [];

  const summaryColumns = useMemo(
    () => [
      { accessorKey: "factor", header: "Factor" },
      {
        accessorKey: "averageExcessReturn",
        header: "Av. Ann. Excess Return (%)",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
        Cell: ({ cell }) => {
          const value = cell.getValue();
          return value === "—" ? "—" : `${value}%`;
        },
      },
      {
        accessorKey: "returnContribution",
        header: "Return Contribution (%)",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
        Cell: ({ cell }) => {
          const value = cell.getValue();
          return value === "—" ? "—" : `${value}%`;
        },
      },
    ],
    [],
  );

  const summaryData = useMemo(
    () =>
      Array.isArray(resultData.summary_table)
        ? resultData.summary_table.map((row) => ({
            factor: row["Factor"],
            averageExcessReturn:
              row["Av. Ann. Excess Return"] !== null && row["Av. Ann. Excess Return"] !== undefined
                ? (parseFloat(row["Av. Ann. Excess Return"]) * 100).toFixed(2)
                : "—",
            returnContribution:
              row["Return Contribution"] !== null && row["Return Contribution"] !== undefined
                ? parseFloat(row["Return Contribution"]).toFixed(2)
                : "—",
          }))
        : [],
    [resultData.summary_table],
  );

  return (
    <Box mt={4}>
      <Typography variant="h6" gutterBottom>Model Output</Typography>

      {(errorMessages.length > 0 || infoMessages.length > 0 || warningMessages.length > 0) && (
        <Box mt={2}>
          <Stack spacing={1}>
            {errorMessages.map((message, idx) => (
              <Alert severity="error" key={`reg-error-inline-${idx}`}>
                {message}
              </Alert>
            ))}
            {infoMessages.map((message, idx) => (
              <Alert severity="info" key={`reg-info-${idx}`}>
                {message}
              </Alert>
            ))}
            {warningMessages.map((message, idx) => (
              <Alert severity="warning" key={`reg-warning-${idx}`}>
                {message}
              </Alert>
            ))}
          </Stack>
        </Box>
      )}

      {resultData.regression_output && resultData.regression_output.text_summary && (
        <Paper style={{ padding: "1rem", marginBottom: "1.5rem" }}>
          <Typography variant="h6">Regression Output Summary</Typography>
          {/* Grid container might not be needed if OlsSummary takes full width */}
          <OlsSummary html={resultData.regression_output.text_summary} />
        </Paper>
      )}

      <Typography variant="h6" mt={4}>Charts</Typography>

      {/* Rolling Alpha and Factor Loadings Plotly Chart */}
      {rollingPlotTraces.length > 0 ? (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Plot
            data={rollingPlotTraces}
            layout={rollingPlotLayout}
            style={{ width: '100%'}}
            useResizeHandler={true}
          />
        </Paper>
      ) : (
        Array.isArray(resultData.image_urls) && resultData.image_urls.length > 0 ? null : <Typography>No rolling plot data available.</Typography>
      )}
      
      {/* Existing logic for other backend-generated images, if any */}
      {Array.isArray(resultData.image_urls) && resultData.image_urls.length > 0 && (
        <Grid container spacing={2}>
          {resultData.image_urls.map((url, idx) => (
            <Grid item xs={12} md={6} key={`img-${idx}`}>
              <img 
                src={`${process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000'}${url}?t=${Date.now()}`} 
                alt={`Regression Chart ${idx}`} 
                style={{ width: "100%", border: "1px solid #ddd"}} 
              />
            </Grid>
          ))}
        </Grid>
      )}

      {/* Summary Table (Return Contribution) */}
      {summaryData.length > 0 && (
        <Box mt={4}>
          <DataTable
            title="Summary Table: Return Contribution"
            titleVariant="h6"
            columns={summaryColumns}
            data={summaryData}
            exportFileName="regression_return_contribution"
          />
        </Box>
      )}
    </Box>
  );
}