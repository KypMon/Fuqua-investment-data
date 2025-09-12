import React, { useEffect } from 'react';
import {
  Typography,
  Box,
  Paper,
  Grid,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody
} from "@mui/material";
import Plot from 'react-plotly.js';

export default function ResultSection({ result }) {
  if (!result) return null;

  // Pick helpers
  const pick = (s, c, fb = null) =>
    result[s] != null
      ? result[s]
      : result[c] != null
      ? result[c]
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
    data: []
  });

  // Portfolio data
  const ef       = block.efficient_frontier    || [];
  const etfPts   = block.etf_points             || [];
  const maxSR    = block.max_sr_point           || { x:0,y:0 };
  const allocStk = block.allocation_stack       || [];
  const weights  = block.weights                || [];
  const pieChart = block.pie_chart              || { labels: [], values: [] };

  // Unique assets & x‐axis for allocation
  const assets = weights.map((w) => w.asset);
  const allocX = allocStk.map((p) => p.x);

  return (
    <Box mt={4}>
      {/* Descriptive stats */}
      <Typography variant="h6">Asset Descriptive Statistics</Typography>
      <Paper sx={{ p:2, mb:3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Asset</TableCell>
              <TableCell align="right">Mean</TableCell>
              <TableCell align="right">Std</TableCell>
              <TableCell align="right">SR</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {descriptiveStats.map((r) => (
              <TableRow key={r.asset}>
                <TableCell>{r.asset}</TableCell>
                <TableCell align="right">{r.mean.toFixed(4)}</TableCell>
                <TableCell align="right">{r.std .toFixed(4)}</TableCell>
                <TableCell align="right">{r.sr  .toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {/* Correlation matrix */}
      <Typography variant="h6">Asset Correlation Matrix</Typography>
      <Paper sx={{ p:2, mb:3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell></TableCell>
              {corrMatrix.columns.map((c) => (
                <TableCell key={c} align="right">{c}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {corrMatrix.data.map((row,i) => (
              <TableRow key={i}>
                <TableCell>{corrMatrix.columns[i]}</TableCell>
                {corrMatrix.columns.map((c) => (
                  <TableCell key={c} align="right">
                    {row[c].toFixed(4)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {/* Frontier */}
      <Typography variant="h6" gutterBottom>
        {normal ? "Standard MV Portfolio" : "Robust MV Portfolio"}
      </Typography>
      <Grid container spacing={12} sx={{ mb:8 }}>
        <Grid item s={12}>
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
      <Grid container spacing={3} sx={{ mb:4 }}>
        {/* weight table */}
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" gutterBottom>
            Max Sharpe Ratio Weights
          </Typography>
          <Paper>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Asset</TableCell>
                  <TableCell align="right">Weight&nbsp;(%)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {weights.map((w) => (
                  <TableRow key={w.asset}>
                    <TableCell>{w.asset}</TableCell>
                    <TableCell align="right">
                      {w.weight.toFixed(2)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Paper>
        </Grid>

        {/* only show pie when no‐short */}
        {short === 0 && pieChart.labels.length > 0 && (
          <Grid item xs={12} md={6}>
            <Plot
              data={[{
                labels: pieChart.labels,
                values: pieChart.values,
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
