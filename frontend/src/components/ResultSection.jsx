import { Typography, Box, Paper, Grid } from "@mui/material";

export default function ResultSection({ result }) {
  if (!result) return null;

  return (
    <Box mt={4}>
      <Typography variant="h6">Console Output</Typography>
      <Paper elevation={1} sx={{ padding: 2, whiteSpace: "pre-wrap", fontFamily: "monospace", marginBottom: 3 }}>
        {result.output_text}
      </Paper>

      <Typography variant="h6">Plots</Typography>
      <Grid container spacing={2}>
        {result.image_urls.map((url, idx) => (
          <Grid item xs={12} md={6} key={idx}>
            <img
              src={`http://localhost:5000${url}?t=${Date.now()}`}
              alt={`Plot ${idx}`}
              style={{ width: "100%", borderRadius: 8 }}
            />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
