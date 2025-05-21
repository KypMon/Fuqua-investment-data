import React from "react";
import {
  Typography,
  Box,
  Paper,
  Table, TableHead, TableRow, TableBody, TableCell,
  Grid
} from "@mui/material";
import Plot from "react-plotly.js";

export default function ResultSection({ result }) {
  if (!result) return null;

  const {
    descriptive_stats,
    correlation_matrix,
    efficient_frontier,
    etf_points,
    max_sr_point,
    allocation_stack,
    robust_weights,
    pie_chart,
    short
  } = result;

  // convert raw pie_chart.values (e.g. [0.587,0.327,0.086]) to percentages
  const pct = pie_chart ? pie_chart.values.map(v => {
    const asPct = v > 1 ? v : v * 100;  
    return `${asPct.toFixed(1)}%`;
  }) : null;


  return (
    <Box mt={4}>
      <Typography variant="h5">Portfolio Results</Typography>

      {/* 1) Descriptive Statistics */}
      <Typography variant="h6" sx={{ mt: 3 }}>Asset Descriptive Statistics</Typography>
      <Paper sx={{ overflowX: "auto", mb: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Asset</TableCell>
              <TableCell align="right">Mean</TableCell>
              <TableCell align="right">Std</TableCell>
              <TableCell align="right">Sharpe</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {descriptive_stats.map(row => (
              <TableRow key={row.asset}>
                <TableCell>{row.asset}</TableCell>
                <TableCell align="right">{row.mean.toFixed(4)}</TableCell>
                <TableCell align="right">{row.std.toFixed(4)}</TableCell>
                <TableCell align="right">{row.sr.toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {/* 2) Correlation Matrix */}
      <Typography variant="h6">Asset Correlation Matrix</Typography>
      <Paper sx={{ overflowX: "auto", mb: 3 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell></TableCell>
              {correlation_matrix.columns.map(col => (
                <TableCell key={col} align="right">{col}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {correlation_matrix.data.map((row, i) => (
              <TableRow key={i}>
                <TableCell>{correlation_matrix.columns[i]}</TableCell>
                {correlation_matrix.columns.map(col => (
                  <TableCell key={col} align="right">
                    {row[col].toFixed(4)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {/* 3) Efficient Frontier */}
      <Typography variant="h6" sx={{ mt: 3 }}>Efficient Frontier</Typography>
      <Plot
        data={[
          {
            x: efficient_frontier.map(p => p.x),
            y: efficient_frontier.map(p => p.y),
            mode: "lines",
            name: "Frontier"
          },
          {
            x: etf_points.map(p => p.x),
            y: etf_points.map(p => p.y),
            mode: "markers+text",
            text: etf_points.map(p => p.label),
            textposition: "top center",
            name: "ETFs"
          },
          {
            x: [max_sr_point.x],
            y: [max_sr_point.y],
            mode: "markers+text",
            text: ["Max SR"],
            marker: { color: "red", size: 12, symbol: "star" },
            name: "Max SR"
          }
        ]}
        layout={{
          width: 700,
          height: 400,
          xaxis: { title: "Std Dev" },
          yaxis: { title: "Annual Return" }
        }}
      />

      {/* 4) Allocation Transition */}
      <Typography variant="h6" sx={{ mt: 3 }}>Allocation Transition</Typography>
      <Plot
        data={Object.keys(allocation_stack[0].allocations).map((asset, i) => ({
          x: allocation_stack.map(p => p.x),
          y: allocation_stack.map(p => p.allocations[asset]),
          stackgroup: "one",
          name: asset
        }))}
        layout={{
          width: 700,
          height: 350,
          xaxis: { title: "Std Dev" },
          yaxis: { title: "Weight" }
        }}
      />

      {/* 5) Robust Max‐Sharpe Weights + Pie */}
      <Typography variant="h6" sx={{mt:4}}>Max Sharpe Ratio Weights</Typography>
      <Paper sx={{p:2, mb:4}}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Asset</TableCell>
              <TableCell align="right">Weight (%)</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {robust_weights.map((lbl,i) => (
              <TableRow key={lbl}>
                <TableCell>{lbl.asset}</TableCell>
                <TableCell align="right">{lbl.weight}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {short === 0 && pie_chart && (
        <>
          <Typography variant="h6" sx={{mb:2}}>Weight Pie Chart</Typography>
          <Plot
            data={[{
              type: 'pie',
              labels: pie_chart.labels,
              values: pie_chart.values,
              textinfo: 'label+percent',
              hole: 0.4
            }]}
            layout={{
              margin: { t: 30, b: 30 },
              showlegend: true
            }}
            style={{ width:'100%', height:400 }}
          />
        </>
      )}
    </Box>
  );
}
