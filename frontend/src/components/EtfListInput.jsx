import React from "react";
import {
  Grid,
  TextField,
  Button,
  Typography,
  Box,
  IconButton
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";

export default function EtfListInput({ etflist, setEtflist }) {
  const handleEtfChange = (index, value) => {
    const updated = [...etflist];
    updated[index] = value;
    setEtflist(updated);
  };

  const addEtfField = () => {
    setEtflist([...etflist, ""]);
  };

  const removeEtfField = (index) => {
    if (etflist.length > 1) {
      const updated = [...etflist];
      updated.splice(index, 1);
      setEtflist(updated);
    }
  };

  return (
    <>
      <Typography variant="subtitle1" gutterBottom>
        ETF List
      </Typography>
      <Grid container spacing={2}>
        {etflist.map((etf, idx) => (
          <Grid item xs={12} sm={6} md={4} key={idx}>
            <Box position="relative">
              <TextField
                label={`ETF ${idx + 1}`}
                value={etf}
                onChange={(e) => handleEtfChange(idx, e.target.value)}
                fullWidth
              />
              {etflist.length > 1 && (
                <IconButton
                  size="small"
                  onClick={() => removeEtfField(idx)}
                  sx={{
                    position: "absolute",
                    top: 4,
                    right: 4,
                    zIndex: 2,
                    backgroundColor: "#fff",
                  }}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              )}
            </Box>
          </Grid>
        ))}
        <Grid item xs={12} sm={6} md={4}>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={addEtfField}
            fullWidth
            sx={{ height: "100%" }}
          >
            Add ETF
          </Button>
        </Grid>
      </Grid>
    </>
  );
}
