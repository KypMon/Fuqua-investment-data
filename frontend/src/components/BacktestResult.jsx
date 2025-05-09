// src/components/BacktestResult.jsx
import React from "react";
import { Box, Typography, Grid, Paper, Table, TableHead, TableRow, TableCell, TableBody } from "@mui/material";

export default function BacktestResult({ result }) {
  if (!result) return null;

  return (
    <Box mt={4}>
      <Typography variant="h5">Backtest Output</Typography>

      <Paper elevation={3} sx={{ padding: 2, marginY: 2, whiteSpace: "pre-line", fontFamily: "monospace" }}>
        {result.output_text}
      </Paper>

      <Typography variant="h6" gutterBottom>Charts</Typography>
      <Grid container spacing={2}>
        {Array.isArray(result.image_urls) && result.image_urls.map((url, idx) => (
            <Grid item xs={12} md={6} key={idx}>
            <img src={`http://localhost:5000${url}?t=${Date.now()}`} alt={`Chart ${idx}`} style={{ width: "100%" }} />
            </Grid>
        ))}
      </Grid>

      {Array.isArray(result.summary_table) && result.summary_table.length > 0 && (
        <>
            <Typography variant="h6" mt={4}>Summary Table</Typography>
            <Table size="small">
            <TableHead>
                <TableRow>
                {Object.keys(result.summary_table[0]).map((col) => (
                    <TableCell key={col}>{col}</TableCell>
                ))}
                </TableRow>
            </TableHead>
            <TableBody>
                {result.summary_table.map((row, idx) => (
                <TableRow key={idx}>
                    {Object.values(row).map((val, i) => (
                    <TableCell key={i}>{val}</TableCell>
                    ))}
                </TableRow>
                ))}
            </TableBody>
            </Table>
        </>
        )}

        {Array.isArray(result.regression_table) && result.regression_table.length > 0 && (
        <>
            <Typography variant="h6" mt={4}>Regression Table</Typography>
            <Table size="small">
            <TableHead>
                <TableRow>
                {Object.keys(result.regression_table[0]).map((col) => (
                    <TableCell key={col}>{col}</TableCell>
                ))}
                </TableRow>
            </TableHead>
            <TableBody>
                {result.regression_table.map((row, idx) => (
                <TableRow key={idx}>
                    {Object.values(row).map((val, i) => (
                    <TableCell key={i}>{val}</TableCell>
                    ))}
                </TableRow>
                ))}
            </TableBody>
            </Table>
        </>
        )}
    </Box>
  );
}