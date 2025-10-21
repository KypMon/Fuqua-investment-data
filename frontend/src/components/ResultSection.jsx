import React, { useMemo } from 'react';
import { Typography, Box, Grid } from "@mui/material";
import Plot from 'react-plotly.js';
import DataTable from "./DataTable";

export default function ResultSection({ result }) {
  const hasResult = Boolean(result);
  const safeResult = result ?? {};

  // Pick helpers
  const pick = (s, c, fb = null) =>
    safeResult[s] != null
      ? safeResult[s]
      : safeResult[c] != null
      ? safeResult[c]
      : fb;

  // Top‐level flags
  const short  = pick("short", "short", 0);
  const normal = pick("normal", "normal", 1);

  // Which block to render
  const standardMv = pick("standard_mv", "standardMv", {});
  const robustMv   = pick("robust_mv",   "robustMv",   {});
  const block      = normal ? standardMv : robustMv;

  // Stats & correlation
  const descriptiveStats = pick("descriptive_stats", "descriptiveStats", []);
  const corrMatrix       = pick("correlation_matrix", "correlationMatrix", {
    columns: [],
    data: [],
  });

  // Portfolio data
  const ef       = block.efficient_frontier    || [];
  const etfPts   = block.etf_points             || [];
  const maxSR    = block.max_sr_point           || { x:0,y:0 };
  const minVar   = block.min_var_point          || { x:0,y:0 };
  const allocStk      = block.allocation_stack       || [];
  const maxSRWeights  = block.max_sr_weights         || block.weights || [];
  const maxSRPie      = block.max_sr_pie_chart       || block.pie_chart || { labels: [], values: [] };
  const minVarWeights = block.min_var_weights        || [];
  const minVarPie     = block.min_var_pie_chart      || { labels: [], values: [] };

  // Unique assets & x-axis for allocation
  const assets = maxSRWeights.map((w) => w.asset);
  const allocX = allocStk.map((p) => p.x);

  const descriptiveColumns = useMemo(() => ([
    { accessorKey: "asset", header: "Asset" },
    {
      accessorKey: "mean",
      header: "Mean",
      muiTableHeadCellProps: { align: "right" },
      muiTableBodyCellProps: { align: "right" },
    },
    {
      accessorKey: "std",
      header: "Std",
      muiTableHeadCellProps: { align: "right" },
      muiTableBodyCellProps: { align: "right" },
    },
    {
      accessorKey: "sr",
      header: "SR",
      muiTableHeadCellProps: { align: "right" },
      muiTableBodyCellProps: { align: "right" },
    },
  ]), []);

  const descriptiveData = useMemo(
    () =>
      descriptiveStats.map((r) => ({
        asset: r.asset,
        mean: typeof r.mean === "number" ? r.mean.toFixed(4) : "N/A",
        std: typeof r.std === "number" ? r.std.toFixed(4) : "N/A",
        sr: typeof r.sr === "number" ? r.sr.toFixed(4) : "N/A",
      })),
    [descriptiveStats],
  );

  const correlationColumns = useMemo(() => {
    if (!Array.isArray(corrMatrix.columns) || corrMatrix.columns.length === 0) {
      return [{ accessorKey: "asset", header: "" }];
    }

    return [
      { accessorKey: "asset", header: "" },
      ...corrMatrix.columns.map((col) => ({
        accessorKey: col,
        header: col,
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      })),
    ];
  }, [corrMatrix.columns]);

  const correlationData = useMemo(() => {
    if (!Array.isArray(corrMatrix.data) || corrMatrix.data.length === 0) return [];

    return corrMatrix.data.map((row, idx) => {
      const record = {
        asset: corrMatrix.columns[idx],
      };

      corrMatrix.columns.forEach((col) => {
        const value = row[col];
        record[col] = typeof value === "number" ? value.toFixed(4) : value ?? "N/A";
      });

      return record;
    });
  }, [corrMatrix.columns, corrMatrix.data]);

  const weightsColumns = useMemo(
    () => [
      { accessorKey: "asset", header: "Asset" },
      {
        accessorKey: "weight",
        header: "Weight (%)",
        muiTableHeadCellProps: { align: "right" },
        muiTableBodyCellProps: { align: "right" },
      },
    ],
    [],
  );

  const maxSRWeightsData = useMemo(
    () =>
      maxSRWeights.map((w) => ({
        asset: w.asset,
        weight: typeof w.weight === "number" ? w.weight.toFixed(2) : "N/A",
      })),
    [maxSRWeights],
  );

  const minVarWeightsData = useMemo(
    () =>
      minVarWeights.map((w) => ({
        asset: w.asset,
        weight: typeof w.weight === "number" ? w.weight.toFixed(2) : "N/A",
      })),
    [minVarWeights],
  );

  if (!hasResult) {
    return null;
  }

  return (
    <Box mt={4}>
      {/* Descriptive stats */}
      {descriptiveData.length > 0 && (
        <DataTable
          title="Asset Descriptive Statistics"
          titleVariant="h6"
          columns={descriptiveColumns}
          data={descriptiveData}
          exportFileName="asset_descriptive_statistics"
        />
      )}

      {/* Correlation matrix */}
      {correlationData.length > 0 && (
        <DataTable
          title="Asset Correlation Matrix"
          titleVariant="h6"
          columns={correlationColumns}
          data={correlationData}
          exportFileName="asset_correlation_matrix"
        />
      )}

      {/* Frontier */}
      <Typography variant="h6" gutterBottom>
        {normal ? "Standard MV Portfolio" : "Robust MV Portfolio"}
      </Typography>
      <Grid container spacing={12} sx={{ mb:8 }}>
        <Grid item xs={12}>
          <Plot
            data={[
              {
                x: ef.map((p) => p.x),
                y: ef.map((p) => p.y),
                mode: 'lines+markers',
                name: 'Eff. Frontier'
              },
              {
                    x: etfPts.map((p) => p.x),
                    y: etfPts.map((p) => p.y),
                    text: etfPts.map((p) => p.label),
                    mode: 'markers+text',
                    name: 'ETFs',
                    textposition: 'top center'
              },
              {
                x: [maxSR.x],
                y: [maxSR.y],
                mode: 'markers+text',
                name: 'Max SR',
                text: ['Max SR'],
                marker: { color:'red', size:12, symbol:'star' }
              },
              {
                x: [minVar.x],
                y: [minVar.y],
                mode: 'markers+text',
                name: 'Min Var',
                text: ['Min Var'],
                marker: { color:'green', size:12, symbol:'star' }
              }
            ]}
            layout={{
              title: normal
                ? 'Standard Efficient Frontier'
                : 'Robust Efficient Frontier',
              xaxis: { title:'Std Dev' },
              yaxis: { title:'Ann. Return' },
              margin: { t:40, b:40, l:40, r:20 }
            }}
            style={{ width:'100%', height:400 }}
          />
        </Grid>
      </Grid>

      {/* Pie + Table row */}
      <Typography variant="subtitle1" gutterBottom>
        Max Sharpe Ratio Portfolio
      </Typography>
      <Grid container spacing={3} sx={{ mb:4 }}>
        {/* weight table */}
        <Grid item xs={12} md={6}>
          {maxSRWeightsData.length > 0 && (
            <DataTable
              title="Max Sharpe Ratio Weights"
              columns={weightsColumns}
              data={maxSRWeightsData}
              exportFileName="max_sharpe_ratio_weights"
            />
          )}
        </Grid>

        {/* only show pie when no-short */}
        {short === 0 && maxSRPie.labels.length > 0 && (
          <Grid item xs={12} md={6}>
            <Plot
              data={[{
                labels: maxSRPie.labels,
                values: maxSRPie.values,
                type: 'pie',
                hole: 0.4
              }]}
              layout={{
                title: 'Max SR Weights (Pie)',
                showlegend: true,
                margin: { t:30, b:30, l:20, r:20 }
              }}
              style={{ width:'100%', height:300 }}
            />
          </Grid>
        )}
      </Grid>

      {minVarWeightsData.length > 0 && (
        <>
          <Typography variant="subtitle1" gutterBottom>
            Minimum Variance Portfolio
          </Typography>
          <Grid container spacing={3} sx={{ mb:4 }}>
            <Grid item xs={12} md={6}>
              <DataTable
                title="Min Variance Weights"
                columns={weightsColumns}
                data={minVarWeightsData}
                exportFileName="minimum_variance_weights"
              />
            </Grid>

            {short === 0 && minVarPie.labels.length > 0 && (
              <Grid item xs={12} md={6}>
                <Plot
                  data={[{
                    labels: minVarPie.labels,
                    values: minVarPie.values,
                    type: 'pie',
                    hole: 0.4
                  }]}
                  layout={{
                    title: 'Min Var Weights (Pie)',
                    showlegend: true,
                    margin: { t:30, b:30, l:20, r:20 }
                  }}
                  style={{ width:'100%', height:300 }}
                />
              </Grid>
            )}
          </Grid>
        </>
      )}

      {/* Allocation Transition (standard only) */}
      {normal && allocStk.length > 0 && (
        <>
          <Typography variant="subtitle1" gutterBottom>
            Allocation Transition
          </Typography>
          <Plot
            data={assets.map((asset) => ({
              x: allocX,
              y: allocStk.map((p) => p.allocations[asset] || 0),
              stackgroup: 'one',
              name: asset
            }))}
            layout={{
              xaxis: { title:'Std Dev' },
              yaxis: { title:'Weight' },
              margin: { t:20, b:40, l:40, r:20 }
            }}
            style={{ width:'100%', height:300 }}
          />
        </>
      )}
    </Box>
  );
}
