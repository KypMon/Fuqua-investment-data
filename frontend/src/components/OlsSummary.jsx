import React, { useCallback, useMemo } from 'react';
import { Button } from '@mui/material';
import '../styles/table.css';

const OlsSummary = ({ html }) => {
  const htmlSummary = useMemo(() => {
    if (typeof html === 'string') {
      return html;
    }
    if (html && typeof html.html === 'string') {
      return html.html;
    }
    return '';
  }, [html]);

  const handleDownload = useCallback(() => {
    if (!htmlSummary) return;

    const blob = new Blob([htmlSummary], { type: 'text/html' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'regression_output_summary.html';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [htmlSummary]);

  if (!htmlSummary) return null;

  return (
    <div>
      <Button variant="outlined" size="small" onClick={handleDownload} sx={{ mb: 2 }}>
        Download Summary
      </Button>
      <div id="ols-summary-container" dangerouslySetInnerHTML={{ __html: htmlSummary }} />
    </div>
  );
};

export default OlsSummary;
