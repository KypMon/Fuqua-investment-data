import React from "react";
import {
  Box,
  Typography,
  Grid,
  Paper,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Alert,
  Stack,
} from "@mui/material";
import Plot from 'react-plotly.js'; // Import Plotly

export default function BacktestResult({ result }) {
  if (!result) return null;

  // --- Prepare data for Plotly charts ---

  // 1. Portfolio Growth Plot
  const portfolioGrowthTraces = Array.isArray(result.portfolio_growth_plot_data)
    ? result.portfolio_growth_plot_data.map(series => ({
        type: 'scatter',
        mode: 'lines',
        name: series.name,
        x: series.dates, // Expects array of date strings 'YYYY-MM-DD'
        y: series.values, // Expects array of numbers
      }))
    : [];

  // 2. Annual Returns Plot (Grouped Bar Chart)
  const annualReturnYears = result.annual_returns_plot_data?.years || [];
  const annualReturnTraces = Array.isArray(result.annual_returns_plot_data?.series)
    ? result.annual_returns_plot_data.series.map(series => ({
        type: 'bar',
        name: series.name,
        x: annualReturnYears, // Uses the common 'years' array
        y: series.returns,    // Expects array of numbers (percentages)
      }))
    : [];

  // 3. Drawdown Plot
  const drawdownTraces = Array.isArray(result.drawdown_plot_data)
    ? result.drawdown_plot_data.map(series => ({
        type: 'scatter',
        mode: 'lines',
        name: series.name,
        x: series.dates, // Expects array of date strings 'YYYY-MM-DD'
        y: series.values, // Expects array of numbers (percentages)
      }))
    : [];

  const infoMessages = Array.isArray(result.messages) ? result.messages : [];
  const warningMessages = Array.isArray(result.warnings) ? result.warnings : [];

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

      {/* {result.output_text && (
        <Paper elevation={3} sx={{ padding: 2, marginY: 2, whiteSpace: "pre-line", fontFamily: "monospace" }}>
          {result.output_text}
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
      {Array.isArray(result.image_urls) && result.image_urls.length > 0 && (
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Other Charts (from backend)
          </Typography>
          <Grid container spacing={2}>
            {result.image_urls.map((url, idx) => (
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
      {Array.isArray(result.portfolio_allocations) && result.portfolio_allocations.length > 0 && (
        result.portfolio_allocations.map((portfolio, pIdx) => (
          portfolio.allocations && portfolio.allocations.length > 0 && (
            <Box key={`alloc-table-${pIdx}`} mt={3}>
              <Typography variant="subtitle1" gutterBottom>
                {portfolio.portfolioName} Allocations
              </Typography>
              <Paper>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Ticker</TableCell>
                      <TableCell align="right">Allocation (%)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolio.allocations.map((alloc, aIdx) => (
                      <TableRow key={aIdx}>
                        <TableCell>{alloc.ticker}</TableCell>
                        <TableCell align="right">
                          {alloc.Allocation !== null && alloc.Allocation !== undefined ? alloc.Allocation.toFixed(2) : 'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>
            </Box>
          )
        ))
      )}

      {/* Performance Summary Table */}
      {Array.isArray(result.summary_table) && result.summary_table.length > 0 && (
        <Box mt={4}>
            <Typography variant="h6" gutterBottom>Performance Summary</Typography>
            <Paper>
              <Table size="small">
              <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    {/* Dynamically generate headers from the keys of the first data row, excluding 'Metric' */}
                    {Object.keys(result.summary_table[0] || {}).filter(key => key !== 'Metric').map(col => (
                        <TableCell key={col} align="right">{col}</TableCell>
                    ))}
                  </TableRow>
              </TableHead>
              <TableBody>
                  {result.summary_table.map((row, idx) => (
                  <TableRow key={idx}>
                      <TableCell>{row.Metric}</TableCell>
                      {Object.keys(row).filter(key => key !== 'Metric').map(colKey => (
                          <TableCell key={colKey} align="right">
                              {typeof row[colKey] === 'number' ? row[colKey].toFixed(4) : (row[colKey] === null ? 'N/A' : row[colKey])}
                          </TableCell>
                      ))}
                  </TableRow>
                  ))}
              </TableBody>
              </Table>
            </Paper>
        </Box>
      )}

      {/* Drawdown Tables */}
      {Array.isArray(result.drawdown_tables) && result.drawdown_tables.length > 0 && (
        result.drawdown_tables.map((tableData, pIdx) => (
          tableData.data && tableData.data.length > 0 && (
            <Box key={`drawdown-detail-table-${pIdx}`} mt={3}>
              <Typography variant="subtitle1" gutterBottom>
                Top 3 Drawdowns: {tableData.portfolioName}
              </Typography>
              <Paper>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      {Object.keys(tableData.data[0]).map(col => (
                        <TableCell key={col} align={typeof tableData.data[0][col] === 'number' ? "right" : "left"}>{col}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {tableData.data.map((row, rIdx) => (
                      <TableRow key={rIdx}>
                        {Object.values(row).map((val, cellIdx) => (
                          <TableCell key={cellIdx} align={typeof val === 'number' ? "right" : "left"}>
                            {typeof val === 'number' ? val.toFixed(4) : (val === null ? 'N/A' : val)}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>
            </Box>
          )
        ))
      )}

      {/* Regression Analysis Tables */}
      {Array.isArray(result.regression_table) && result.regression_table.length > 0 && (
         result.regression_table.map((regData, pIdx) => (
            regData.coefficients && regData.coefficients.length > 0 && (
                <Box key={`reg-summary-table-${pIdx}`} mt={3}>
                    <Typography variant="subtitle1" gutterBottom>
                        Regression Analysis vs Benchmark: {regData.portfolioName}
                    </Typography>
                    <Typography variant="body2" sx={{mb:1}}>
                        R-squared: {regData.r_squared !== null ? regData.r_squared.toFixed(4) : 'N/A'}, Adj. R-squared: {regData.adj_r_squared !== null ? regData.adj_r_squared.toFixed(4) : 'N/A'}, Annualized Alpha: {regData.annualized_alpha !== null ? regData.annualized_alpha.toFixed(4) : 'N/A'}
                    </Typography>
                    <Paper>
                        <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Factor</TableCell>
                                <TableCell align="right">Loadings</TableCell>
                                <TableCell align="right">Std. Errors</TableCell>
                                <TableCell align="right">t-stat</TableCell>
                                <TableCell align="right">p-value</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {regData.coefficients.map((row, idx) => (
                            <TableRow key={idx}>
                                <TableCell>{row.Factor}</TableCell>
                                <TableCell align="right">{row.Loadings !== null ? row.Loadings.toFixed(4) : 'N/A'}</TableCell>
                                <TableCell align="right">{row['Standard Errors'] !== null ? row['Standard Errors'].toFixed(4) : 'N/A'}</TableCell>
                                <TableCell align="right">{row['t-stat'] !== null ? row['t-stat'].toFixed(4) : 'N/A'}</TableCell>
                                <TableCell align="right">{row['p-value'] !== null ? row['p-value'].toFixed(4) : 'N/A'}</TableCell>
                            </TableRow>
                            ))}
                        </TableBody>
                        </Table>
                    </Paper>
                </Box>
            )
         ))
      )}
    </Box>
  );
}