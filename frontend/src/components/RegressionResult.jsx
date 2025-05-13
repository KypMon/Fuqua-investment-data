import React from "react";
import {
  Box,
  Typography,
  Paper,
  Grid,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody
} from "@mui/material";

import OlsSummary from './OlsSummary';

export default function RegressionResult({ result }) {
  
  if (!result) return null;

  const resultData = typeof result === "string"
  ? JSON.parse(result)
  : result;

  return (
    <Box mt={4}>
      <Typography variant="h6" gutterBottom>Model Output</Typography>

      <Paper style={{ padding: "1rem", marginBottom: "1.5rem" }}>
        <Typography variant="h6">Regression Output Summary</Typography>
        <Grid container spacing={2}>
          {/* {Object.entries(resultData.regression_output).map(([key, value]) => (
            <Grid item xs={6} md={3} key={key}>
              <Typography><strong>{key.replace(/_/g, " ")}:</strong> {value}</Typography>
            </Grid>
          ))} */}
          {/* <div dangerouslySetInnerHTML={{ __html: resultData.regression_output.text_summary }} /> */}
           <OlsSummary html={resultData.regression_output.text_summary} />
        </Grid>
      </Paper>

      <Typography variant="h6" mt={4}>Charts</Typography>
      <Grid container spacing={2}>
        {resultData.image_urls.map((url, idx) => (
          <Grid item xs={12} md={6} key={idx}>
            <img src={`${process.env.REACT_APP_API_BASE_URL}${url}`} alt={`Regression Chart ${idx}`} width="100%" />
          </Grid>
        ))}
      </Grid>

      {resultData.summary_table?.length > 0 && (
        <>
          <Typography variant="h6" gutterBottom>Summary Table</Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Factor</TableCell>
                  <TableCell>Av. Ann. Excess Return</TableCell>
                  <TableCell>Return Contribution (%)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {resultData.summary_table.map((row, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{row["Factor"]}</TableCell>
                    <TableCell>{(row["Av. Ann. Excess Return"] * 100).toFixed(2)}%</TableCell>
                    <TableCell>
                      {row["Return Contribution"] != null
                        ? `${row["Return Contribution"].toFixed(2)}%`
                        : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
        </>
      )}
    </Box>
  );
}
